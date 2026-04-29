"""Feynman diagram representation using networkx.

Each diagram is a MultiGraph where:
- Nodes are either external points (observable fields) or interaction vertices
- Edges are propagators (C or R)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import permutations, product as iter_product
from typing import Sequence

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
        full_label: str | None = None,
    ) -> str:
        """Add an external point (observable field) to the diagram.

        Args:
            label: Default display label for the node (typically a
                compact form like ``"$\\phi_a$"``).
            field_type: ``"physical"`` or ``"response"``.
            component: Component index (``None`` for scalar fields).
            spatial: Spatial argument string.
            full_label: Optional richer label that includes the
                spatial argument (e.g. ``"$\\phi_a(x_1)$"``).
                Stashed under the ``full_label`` node attribute so
                renderers can opt into it via the ``LABEL_FULL``
                format flag.  When ``None`` no ``full_label`` key is
                set (renderers fall back to ``label``).

        Returns:
            The unique node ID assigned to this external point.
        """
        node_id = f"ext_{self._node_counter}"
        self._node_counter += 1
        node_data: dict[str, object] = dict(
            node_type="external",
            label=label,
            field_type=field_type,
            component=component,
            spatial=spatial,
        )
        if full_label is not None:
            node_data["full_label"] = full_label
        self.graph.add_node(node_id, **node_data)
        return node_id

    def add_vertex(
        self,
        coupling: str,
        copy_id: int = 0,
        spatial_vars: Sequence[str] = (),
    ) -> str:
        """Add an interaction vertex to the diagram.

        Args:
            coupling: Coupling constant name (e.g. ``"F"``, ``"g"``).
            copy_id: Which copy of the vertex in the expansion.
            spatial_vars: Spatial integration variables for this vertex.

        Returns:
            The unique node ID assigned to this vertex.
        """
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
        phi_end: str | None = None,
        psi_end: str | None = None,
    ) -> None:
        """Add a propagator edge between two nodes.

        Args:
            node1: Source node ID.
            node2: Target node ID.
            kind: ``"C"`` for correlation or ``"R"`` for response.
            index_left: Left component index (``None`` for scalars).
            index_right: Right component index (``None`` for scalars).
            spatial_left: Left spatial argument.
            spatial_right: Right spatial argument.
            phi_end: For R propagators, the node ID on the physical
                (φ) side.  ``None`` for C propagators.
            psi_end: For R propagators, the node ID on the response
                (ψ) side.  ``None`` for C propagators.
        """
        self.graph.add_edge(
            node1,
            node2,
            kind=kind,
            index_left=index_left,
            index_right=index_right,
            spatial_left=spatial_left,
            spatial_right=spatial_right,
            phi_end=phi_end,
            psi_end=psi_end,
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
        """Construct a diagram from a Wick contraction pairing.

        Args:
            observable_ops: External field operators.
            vertex_instances: Instantiated interaction vertices.
            pairing: Tuple of ``(i, j)`` index pairs from Wick
                contraction.

        Returns:
            A fully-constructed ``FeynmanDiagram`` with external nodes,
            vertex nodes, and propagator edges.
        """
        diagram = cls()

        # Build the full operator list (same order as in wick contraction)
        all_ops: list[FieldOperator] = list(observable_ops)
        for vi in vertex_instances:
            all_ops.extend(vi.field_operators)

        # Map operator UID -> graph node ID
        uid_to_node: dict[int, str] = {}

        # Add external nodes.  The displayed label is *compact* by
        # default ($\phi_a$); the full form ($\phi_a(x_1)$) is stashed
        # in ``full_label`` so the rendering layer can opt back into
        # it via ``label_format=LABEL_FULL``.  See render_labels.py.
        for op in observable_ops:
            name = r"\phi" if op.is_physical else r"\psi"
            if op.component_index is not None:
                compact_label = rf"${name}_{{{op.component_index}}}$"
                full_label = rf"${name}_{{{op.component_index}}}({op.spatial_arg})$"
            else:
                compact_label = f"${name}$"
                full_label = rf"${name}({op.spatial_arg})$"

            node_id = diagram.add_external_point(
                label=compact_label,
                field_type=op.field_type.value,
                component=op.component_index,
                spatial=op.spatial_arg,
                full_label=full_label,
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
                if op.uid in uid_to_node:
                    raise ValueError(
                        f"UID collision: vertex operator {op} has the same "
                        f"uid={op.uid} as a previously registered operator. "
                        f"Call reset_uid_counter() before creating fields, "
                        f"not between field creation and compute_moment()."
                    )
                uid_to_node[op.uid] = node_id

        # Add edges for each contraction pair
        for i, j in pairing:
            op_i, op_j = all_ops[i], all_ops[j]
            prop = contract_pair(op_i, op_j)
            if prop is not None:
                node_a = uid_to_node[op_i.uid]
                node_b = uid_to_node[op_j.uid]

                # Track R-propagator direction (which end is φ, which is ψ)
                phi_end = None
                psi_end = None
                if prop.kind == "R":
                    if op_i.is_physical:
                        phi_end, psi_end = node_a, node_b
                    else:
                        phi_end, psi_end = node_b, node_a

                diagram.add_propagator(
                    node_a,
                    node_b,
                    kind=prop.kind,
                    index_left=prop.index_left,
                    index_right=prop.index_right,
                    spatial_left=prop.spatial_left,
                    spatial_right=prop.spatial_right,
                    phi_end=phi_end,
                    psi_end=psi_end,
                )

        return diagram

    def canonical_form(self) -> tuple:
        """Return a hashable canonical form for this diagram's topology.

        Two diagrams have the same canonical form if and only if they
        are isomorphic under relabeling of vertex nodes that share the
        same coupling type.  External nodes are distinguished by their
        position in the observable.  Edge kind (C/R) and R-direction
        are structural; component indices and spatial arguments on
        edges are ignored.

        Returns:
            A hashable tuple ``(ext_meta, vert_meta, edges)`` that is
            identical for topologically equivalent diagrams.
        """
        g = self.graph
        ext_nodes = self.external_nodes
        vert_nodes = self.vertex_nodes

        # External nodes are distinguishable — label them 0..N-1
        ext_label: dict[str, int] = {}
        for i, n in enumerate(ext_nodes):
            ext_label[n] = i

        # Group vertex nodes by coupling type
        coupling_groups: dict[str, list[str]] = defaultdict(list)
        for n in vert_nodes:
            coupling_groups[g.nodes[n]["coupling"]].append(n)

        sorted_couplings = sorted(coupling_groups.keys())
        group_lists = [coupling_groups[c] for c in sorted_couplings]

        # Try all permutations within each coupling group
        perm_generators = [permutations(grp) for grp in group_lists]

        best: tuple | None = None

        for perm_combo in iter_product(*perm_generators):
            label_map = dict(ext_label)
            counter = len(ext_nodes)
            for group_perm in perm_combo:
                for node in group_perm:
                    label_map[node] = counter
                    counter += 1

            edges: list[tuple] = []
            for u, v, data in g.edges(data=True):
                kind = data["kind"]
                if kind == "C":
                    lu, lv = label_map[u], label_map[v]
                    edges.append(("C", min(lu, lv), max(lu, lv)))
                else:
                    # R is directed: use phi_end / psi_end if available
                    pe = data.get("phi_end")
                    se = data.get("psi_end")
                    if pe is not None and se is not None:
                        edges.append(("R", label_map[pe], label_map[se]))
                    else:
                        # Fallback: use storage order
                        edges.append(("R", label_map[u], label_map[v]))

            form = tuple(sorted(edges))
            if best is None or form < best:
                best = form

        # If no vertices (and thus no permutations to try), handle the
        # trivial case where iter_product produces exactly one empty combo
        if best is None:
            label_map = dict(ext_label)
            edges = []
            for u, v, data in g.edges(data=True):
                kind = data["kind"]
                lu, lv = label_map.get(u, -1), label_map.get(v, -1)
                if kind == "C":
                    edges.append(("C", min(lu, lv), max(lu, lv)))
                else:
                    pe = data.get("phi_end")
                    se = data.get("psi_end")
                    if pe is not None and se is not None:
                        edges.append(("R", label_map.get(pe, -1), label_map.get(se, -1)))
                    else:
                        edges.append(("R", lu, lv))
            best = tuple(sorted(edges))

        ext_meta = tuple(
            (g.nodes[n]["field_type"], g.nodes[n].get("component"), g.nodes[n]["spatial"])
            for n in ext_nodes
        )
        vert_meta = tuple((c, len(coupling_groups[c])) for c in sorted_couplings)

        return (ext_meta, vert_meta, best)

    def summary(self, short: bool = False) -> str:
        """Short textual description of the diagram topology.

        Args:
            short: If ``True``, return a compact single-line summary
                suitable for subplot titles.
        """
        n_ext = len(self.external_nodes)
        n_vert = len(self.vertex_nodes)
        n_edges = self.graph.number_of_edges()
        loops = self.n_loops
        if short:
            conn = "conn" if self.is_connected else "disc"
            return f"{n_ext}ext, {n_vert}vert, {n_edges}prop, {loops}L, {conn}"
        conn = "connected" if self.is_connected else "disconnected"
        return f"Diagram: {n_ext} external, {n_vert} vertices, {n_edges} propagators, {loops} loops, {conn}"
