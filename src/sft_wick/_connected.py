"""Occurrence incidence for the optional connected-diagram target.

Coordinates and component indices are labels of slots, never statistical
connections. Each external occurrence owns a node and each vertex instance
owns one node, including all legs of a nonlocal input-cumulant vertex.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .expressions import Propagator
    from .fields import FieldOperator
    from .vertices import Vertex, VertexInstance
    from .wick import Pairing, SpatialSignature


def validate_target(target: str, observable: Sequence[object]) -> None:
    """Validate the target without changing the legacy moment contract."""
    if target not in ("moment", "connected"):
        raise ValueError("target must be 'moment' or 'connected'")
    if target == "connected" and not observable:
        raise ValueError(
            "target='connected' requires at least one observable occurrence"
        )


def connected_sequence_possible(n_observable: int, vertices: Sequence[Vertex]) -> bool:
    """A necessary edge-count condition, before any Wick enumeration.

    A pairing has E=(q+sum(m_v))/2 edges on q+n incidence nodes. A connected
    graph requires E >= q+n-1. A vertex without slots is isolated.
    """
    sizes = tuple(vertex.n_fields for vertex in vertices)
    total = n_observable + sum(sizes)
    return all(sizes) and total % 2 == 0 and total // 2 >= n_observable + len(sizes) - 1


@dataclass(frozen=True)
class ConnectedIncidence:
    """The fixed owner of every operator-list occurrence.

    ``disjoint_labels`` proves that component routing at a spatial label
    cannot move a slot to another owner. If false, connectivity must be
    checked on every routed occurrence pairing, not its representative.
    """

    owners: tuple[int, ...]
    n_nodes: int
    disjoint_labels: bool

    @classmethod
    def from_instances(
        cls,
        observable: Sequence[FieldOperator],
        vertex_instances: Sequence[VertexInstance],
    ) -> ConnectedIncidence:
        owners = list(range(len(observable)))
        operators = list(observable)
        for k, vi in enumerate(vertex_instances, len(observable)):
            operators.extend(vi.field_operators)
            owners.extend([k] * len(vi.field_operators))
        label_owner: dict[str | None, int] = {}
        disjoint = True
        for op, owner in zip(operators, owners):
            old = label_owner.setdefault(op.spatial_arg, owner)
            if old != owner:
                disjoint = False
        return cls(tuple(owners), len(observable) + len(vertex_instances), disjoint)

    def is_connected(self, pairing: Pairing) -> bool:
        """Require one component including all external and vertex nodes."""
        if self.n_nodes == 0:
            return False
        parent = list(range(self.n_nodes))

        def find(node):
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        components = self.n_nodes
        for left, right in pairing:
            a, b = find(self.owners[left]), find(self.owners[right])
            if a != b:
                parent[a] = b
                components -= 1
        return components == 1

    def prune_topologies(
        self,
        spatial_results: dict[SpatialSignature, tuple[list[Propagator], int, Pairing]],
    ) -> dict[SpatialSignature, tuple[list[Propagator], int, Pairing]]:
        """Prune representatives only when ownership invariance is proved."""
        if not self.disjoint_labels:
            return spatial_results
        return {
            sig: record
            for sig, record in spatial_results.items()
            if self.is_connected(record[2])
        }

    def select_routings(
        self,
        routings: list[tuple[list[Propagator], Pairing]],
    ) -> list[tuple[list[Propagator], Pairing]]:
        """Exact fallback for labels shared by different slot owners."""
        if self.disjoint_labels:
            return routings
        return [record for record in routings if self.is_connected(record[1])]
