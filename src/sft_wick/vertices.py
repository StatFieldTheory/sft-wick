"""Vertex definitions and vertex instantiation.

A Vertex is a template for an interaction term in S_int.
A VertexInstance is a concrete instantiation with freshly assigned indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from numbers import Integral, Real
from typing import Iterable, Sequence

from ._util import require_bool
from .expressions import Symbol
from .fields import Field, FieldOperator
from .indices import IndexContext


LEG_SYMMETRIES = ("none", "declared", "symmetrize")


def validate_leg_symmetry(value: str) -> None:
    if not isinstance(value, str) or value not in LEG_SYMMETRIES:
        raise ValueError(
            "leg_symmetry must be 'none', 'declared', or 'symmetrize'"
        )


def equal_time_aliases(legs: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """``(leg, first leg)`` for every leg after the first of an equal-time
    group: the first leg created stands for the group's one time."""
    return tuple((leg, legs[0]) for leg in legs[1:])


def normalize_leg_groups(
    n_fields: int, leg_groups, group_mode: str, group_lags,
) -> tuple[tuple[int, ...], tuple[tuple[int, int, float], ...]]:
    """Validate ordered field-to-group assignments and flatten lag forests.

    Each lag (child, parent, d) means t_child = t_parent - d.
    Independent groups have no parent. Flattening makes the field-to-time
    map independent of the input ordering of the relations.
    """
    if group_mode not in ("spacetime", "time"):
        raise ValueError("group_mode must be 'spacetime' or 'time'")
    groups = tuple(leg_groups)
    lags = tuple(group_lags)
    if not groups:
        if lags or group_mode != "spacetime":
            raise ValueError("group_lags and group_mode='time' require leg_groups")
        return (), ()
    if len(groups) != n_fields:
        raise ValueError("leg_groups must assign one group to each field slot")
    if any(not isinstance(g, Integral) or isinstance(g, bool) for g in groups):
        raise TypeError("leg_groups must contain integer group IDs")
    groups = tuple(map(int, groups))
    count = len(set(groups))
    if set(groups) != set(range(count)):
        raise ValueError("leg_groups IDs must run from zero to the number of groups minus one")
    parents = {}
    for relation in lags:
        if not isinstance(relation, (list, tuple)) or len(relation) != 3:
            raise ValueError("group_lags entries must be (delayed_group, anchor_group, lag)")
        child, parent, lag = relation
        if any(not isinstance(g, Integral) or isinstance(g, bool)
               or not 0 <= g < count for g in (child, parent)):
            raise ValueError("group_lags references an invalid group ID")
        if not isinstance(lag, Real) or isinstance(lag, bool) or not isfinite(lag) or lag < 0:
            raise ValueError("group_lags delays must be finite nonnegative real numbers")
        if child in parents:
            raise ValueError("group_lags assigns multiple anchors to one group")
        if isinstance(lag, Integral):
            exact = Fraction(int(lag))
        else:
            numerator, denominator = lag.as_integer_ratio()
            exact = Fraction(int(numerator), int(denominator))
        parents[int(child)] = (int(parent), exact)
    return groups, flatten_group_lags(parents)


def flatten_group_lags(parents):
    """Return (child, root, binary64 delay) for a forest of exact offsets.

    Each node has at most one parent and its edge delay is a nonnegative
    Fraction. Nodes are mutually sortable labels, such as the integer
    groups of a vertex or the string coordinates of a diagram.
    """
    flattened = []
    clock_values = {}
    precision_error = (
        "group_lags exceeds floating-point precision for these relative delays; "
        "choose an anchor closer to the delayed times or reformulate their scales"
    )
    for child in sorted(parents):
        seen = set()
        node, delay = child, Fraction(0)
        while node in parents:
            if node in seen:
                raise ValueError("group_lags contains a cycle")
            seen.add(node)
            node, edge = parents[node]
            total = delay + edge
            try:
                rounded = float(total)
            except OverflowError:
                raise ValueError("group_lags accumulated delay must be finite") from None
            if (total and rounded == 0 or
                    delay and edge and rounded == max(float(delay), float(edge))):
                raise ValueError(precision_error)
            delay = total
        # Different branches can also land on the same representable offset.
        # Equal rounded times must never turn distinct clocks into an Itô alias.
        if clock_values.setdefault((node, rounded), delay) != delay:
            raise ValueError(precision_error)
        flattened.append((child, node, rounded))
    return tuple(flattened)


@dataclass(frozen=True)
class Vertex:
    """A vertex (interaction term) in S_int.

    Examples:
        # Local vertex: int F_{ijk} phi_i(x) phi_j(x) psi_k(x) dx
        Vertex(fields=[phi, phi, psi], coupling='F')

        # Non-local vertex: iint K_{ij}(x,x') psi_i(x) psi_j(x') dx dx'
        Vertex(fields=[psi, psi], coupling='K', local=False)

        # One bilocal kernel evaluated at two spacetime points.
        Vertex([phi, phi, psi, psi], 'K', local=False, leg_groups=(0, 1, 0, 1))

    In ``group_mode="spacetime"``, each group contributes one coordinate
    argument to the coupling, in ascending group ID order. Mode ``"time"``
    identifies the clocks within each group and leaves one spatial argument
    per field slot, in field order. Tensor indices follow the field slots
    in both modes. A ``(child, parent, lag)`` entry in ``group_lags`` ties
    the child time to the parent time minus the nonnegative lag.
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
    leg_symmetry: str = "none"
    leg_groups: tuple[int, ...] = ()
    group_mode: str = "spacetime"
    group_lags: tuple[tuple[int, int, float], ...] = ()

    def __init__(
        self,
        fields: Sequence[Field],
        coupling: str,
        local: bool = True,
        equal_time: bool = False,
        already_R_contracted: bool = False,
        leg_symmetry: str = "none",
        *,
        leg_groups: Sequence[int] = (),
        group_mode: str = "spacetime",
        group_lags: Sequence[tuple[int, int, float]] = (),
    ):
        validate_leg_symmetry(leg_symmetry)
        if local and leg_symmetry != "none":
            raise ValueError("leg_symmetry is only set explicitly on non-local vertices")
        equal_time = require_bool(equal_time, "equal_time", "Vertex")
        already_R_contracted = require_bool(
            already_R_contracted, "already_R_contracted", "Vertex")
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
        fields = tuple(fields)
        leg_groups, group_lags = normalize_leg_groups(
            len(fields), leg_groups, group_mode, group_lags
        )
        if leg_groups and (local or equal_time or already_R_contracted):
            raise ValueError("leg_groups requires a non-local vertex with equal_time=False "
                             "and already_R_contracted=False")
        if already_R_contracted and any(f.is_physical for f in fields):
            raise ValueError(
                "already_R_contracted=True takes response fields only: the "
                "callable has integrated each leg against the R that joins "
                "it to its partner, and a phi leg has no such R of its own."
            )
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "coupling", coupling)
        object.__setattr__(self, "local", local)
        object.__setattr__(self, "equal_time", bool(equal_time))
        object.__setattr__(self, "already_R_contracted", bool(already_R_contracted))
        object.__setattr__(self, "leg_symmetry", leg_symmetry)
        object.__setattr__(self, "leg_groups", leg_groups)
        object.__setattr__(self, "group_mode", group_mode)
        object.__setattr__(self, "group_lags", group_lags)

    @property
    def slot_points(self) -> tuple[int, ...]:
        """Coordinate index of each field in the ordered coupling arguments."""
        if self.local:
            return (0,) * len(self.fields)
        if self.leg_groups and self.group_mode == "spacetime":
            return self.leg_groups
        return tuple(range(len(self.fields)))

    def time_relations(self, points):
        """Zero-lag aliases and positive shifts on this copy's point labels."""
        if not self.leg_groups:
            return (equal_time_aliases(points) if self.equal_time else ()), ()
        point_groups = (self.leg_groups if self.group_mode == "time"
                        else range(len(points)))
        roots = {g: (root, d) for g, root, d in self.group_lags}
        times = [roots.get(g, (g, 0.0)) for g in point_groups]
        representatives = {}
        aliases = []
        for point, key in zip(points, times, strict=True):
            rep = representatives.setdefault(key, point)
            if point != rep:
                aliases.append((point, rep))
        shifts = tuple((point, representatives[root, 0.0], delay)
                       for (root, delay), point in representatives.items() if delay)
        return tuple(aliases), shifts

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
    time_lags: tuple[tuple[str, str, float], ...] = ()

    @classmethod
    def instantiate(
        cls,
        vertex: Vertex,
        idx_ctx: IndexContext,
        copy_id: int,
    ) -> VertexInstance:
        """Create a concrete vertex instance with fresh indices.

        A local vertex has one spatial variable. A grouped spacetime vertex
        has one per group. All other non-local vertices have one per field.
        """
        slot_points = vertex.slot_points
        n_points = 1 if vertex.local else len(set(slot_points))
        spatial_vars = [idx_ctx.fresh_spatial_variable() for _ in range(n_points)]
        operators = []
        comp_indices = []
        for f, coordinate in zip(vertex.fields, slot_points, strict=True):
            point = spatial_vars[coordinate]
            if f.is_scalar:
                operators.append(f(point))
            else:
                component = idx_ctx.fresh_component_index()
                comp_indices.append(component)
                operators.append(f(component, point))

        # Recording even the single local coordinate distinguishes vertex
        # copies for a time-dependent coupling; local=True suppresses its
        # argument in the traditional printed expression.
        coupling_symbol = Symbol(vertex.coupling, tuple(comp_indices), tuple(spatial_vars),
                                 local=vertex.local)
        aliases, shifts = vertex.time_relations(spatial_vars)
        return cls(
            vertex=vertex,
            field_operators=operators,
            coupling_symbol=coupling_symbol,
            spatial_variables=spatial_vars,
            component_indices=comp_indices,
            copy_id=copy_id,
            equal_time_aliases=aliases,
            time_lags=shifts,
        )


def all_operators(
    observable: Iterable[FieldOperator],
    vertex_instances: Iterable[VertexInstance],
) -> list[FieldOperator]:
    """The operator list a pairing indexes: the observable, then the
    operators of each vertex instance."""
    operators = list(observable)
    for vi in vertex_instances:
        operators.extend(vi.field_operators)
    return operators
