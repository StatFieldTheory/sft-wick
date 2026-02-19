"""Feynman diagram representation using networkx.

Each diagram is a MultiGraph where:
- Nodes are either external points (observable fields) or interaction vertices
- Edges are propagators (C or R)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import networkx as nx

from .fields import FieldOperator
from .propagators import contract_pair
from .vertices import VertexInstance
from .wick import Pairing


@dataclass
class FeynmanDiagram:
    """Graph-based representation of a single Feynman diagram."""

    graph: nx.MultiGraph = field(default_factory=nx.MultiGraph)
    _node_counter: int = field(default=0, repr=False)

    def add_external_point(
        self,
        label: str,
        field_type: str,
        component: str | None = None,
        spatial: str = "",
    ) -> str:
        """Add an external point (observable field) to the diagram."""
        node_id = f"ext_{self._node_counter}"
        self._node_counter += 1
        self.graph.add_node(
            node_id,
            node_type="external",
            label=label,
            field_type=field_type,
            component=component,
            spatial=spatial,
        )
        return node_id

    def add_vertex(
        self,
        coupling: str,
        copy_id: int = 0,
        spatial_vars: Sequence[str] = (),
    ) -> str:
        """Add an interaction vertex to the diagram."""
        node_id = f"vert_{self._node_counter}"
        self._node_counter += 1
        self.graph.add_node(
            node_id,
            node_type="vertex",
            label=coupling,
            coupling=coupling,
            copy_id=copy_id,
            spatial_vars=list(spatial_vars),
        )
        return node_id

    def add_propagator(
        self,
        node1: str,
        node2: str,
        kind: str,
        index_left: str | None = None,
        index_right: str | None = None,
        spatial_left: str = "",
        spatial_right: str = "",
    ) -> None:
        """Add a propagator edge between two nodes."""
        self.graph.add_edge(
            node1,
            node2,
            kind=kind,
            index_left=index_left,
            index_right=index_right,
            spatial_left=spatial_left,
            spatial_right=spatial_right,
        )

    @property
    def external_nodes(self) -> list[str]:
        return [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "external"]

    @property
    def vertex_nodes(self) -> list[str]:
        return [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "vertex"]

    @property
    def n_loops(self) -> int:
        """Number of loops = E - V + connected_components."""
        e = self.graph.number_of_edges()
        v = self.graph.number_of_nodes()
        c = nx.number_connected_components(self.graph)
        return e - v + c

    @property
    def is_connected(self) -> bool:
        if self.graph.number_of_nodes() == 0:
            return True
        return nx.is_connected(self.graph)

    @classmethod
    def from_pairing(
        cls,
        observable_ops: list[FieldOperator],
        vertex_instances: list[VertexInstance],
        pairing: Pairing,
    ) -> FeynmanDiagram:
        """Construct a diagram from a Wick contraction pairing."""
        diagram = cls()

        # Build the full operator list (same order as in wick contraction)
        all_ops: list[FieldOperator] = list(observable_ops)
        for vi in vertex_instances:
            all_ops.extend(vi.field_operators)

        # Map operator UID -> graph node ID
        uid_to_node: dict[int, str] = {}

        # Add external nodes
        for op in observable_ops:
            node_id = diagram.add_external_point(
                label=repr(op),
                field_type=op.field_type.value,
                component=op.component_index,
                spatial=op.spatial_arg,
            )
            uid_to_node[op.uid] = node_id

        # Add vertex nodes (one per vertex instance, not per operator)
        vi_to_node: dict[int, str] = {}
        for vi in vertex_instances:
            node_id = diagram.add_vertex(
                coupling=vi.vertex.coupling,
                copy_id=vi.copy_id,
                spatial_vars=vi.spatial_variables,
            )
            vi_to_node[vi.copy_id] = node_id
            for op in vi.field_operators:
                uid_to_node[op.uid] = node_id

        # Add edges for each contraction pair
        for i, j in pairing:
            op_i, op_j = all_ops[i], all_ops[j]
            prop = contract_pair(op_i, op_j)
            if prop is not None:
                node_a = uid_to_node[op_i.uid]
                node_b = uid_to_node[op_j.uid]
                diagram.add_propagator(
                    node_a,
                    node_b,
                    kind=prop.kind,
                    index_left=prop.index_left,
                    index_right=prop.index_right,
                    spatial_left=prop.spatial_left,
                    spatial_right=prop.spatial_right,
                )

        return diagram

    def summary(self) -> str:
        """Short textual description of the diagram topology."""
        n_ext = len(self.external_nodes)
        n_vert = len(self.vertex_nodes)
        n_edges = self.graph.number_of_edges()
        loops = self.n_loops
        conn = "connected" if self.is_connected else "disconnected"
        return f"Diagram: {n_ext} external, {n_vert} vertices, {n_edges} propagators, {loops} loops, {conn}"
