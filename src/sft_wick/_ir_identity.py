"""Typed spatial identities of the topologies represented by a DiagramTerm."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import TYPE_CHECKING

from ._canonical_search import PointGraph, Search
from ._shifted_times import normalize_lags
from .expressions import IntegralOver, Product, Sum, SumOverIndex, Symbol

if TYPE_CHECKING:
    from .perturbation import DiagramTerm


_ID_SCHEMA = "wickir-typed-graph/v1"


def _coupling_layouts(expr, represented):
    """Distinct monomials of typed coupling occurrences, modulo routing.

    Deduplicate each factor before multiplying layouts. Many components
    can route through the exact same typed spatial graph.
    """
    if isinstance(expr, Symbol):
        if (expr.name, frozenset(expr.spatial_args)) in represented:
            return {()}
        return {((expr.name, expr.local, len(expr.indices), expr.spatial_args, ()),)}
    if isinstance(expr, Sum):
        return {layout for child in expr.terms for layout in _coupling_layouts(child, represented)}
    if isinstance(expr, Product):
        layouts = {()}
        for factor in expr.factors:
            right_layouts = _coupling_layouts(factor, represented)
            layouts = {tuple(sorted(left + right)) for left in layouts
                       for right in right_layouts}
        return layouts
    if isinstance(expr, (IntegralOver, SumOverIndex)):
        return _coupling_layouts(expr.body, represented)
    return {()}


def _time_groups(aliases):
    """Equal-time classes are invariant to the choice of representative."""
    parents = {point: point for edge in aliases for point in edge}

    def root(point):
        while parents[point] != point:
            parents[point] = parents[parents[point]]
            point = parents[point]
        return point

    for point, rep in aliases:
        parents[root(point)] = root(rep)
    groups = defaultdict(list)
    for point in parents:
        groups[root(point)].append(point)
    return [group for group in groups.values() if len(group) > 1]


def _typed_graph_id(term, layout):
    integrated = set(term.integration_vars)
    points = integrated.union(
        endpoint for prop in term.propagators
        for endpoint in (prop.spatial_left, prop.spatial_right)
    )
    points.update(point for _, _, _, args, _ in layout for point in args)
    points.update(point for pair in (*term.equal_time_aliases, *term.r_absorbed_pairs)
                  for point in pair)
    points.update(point for delayed, anchor, _ in term.time_lags for point in (delayed, anchor))
    labels = sorted(points)
    node = {point: i for i, point in enumerate(labels)}
    colours = [("internal",) if point in integrated else ("external", point)
               for point in labels]

    # Each coupling owns explicit, ordered slots. The slot nodes keep the
    # incidence unambiguous even when two slots name the same point.
    slots = []
    instances = []
    for owner, (_, _, _, args, _) in enumerate(layout):
        legs = []
        for position, point in enumerate(args):
            slot = len(labels)
            labels.append((owner, position))
            colours.append(("slot", position))
            legs.append(slot)
            slots.append((slot, node[point]))
        instances.append(legs)

    for name, local, rank, args, template in layout:
        # v1.1 layouts may combine a full source template and a layout
        # reconstructed from an expression. JSON keeps their sort order total.
        identity = json.dumps(template, separators=(",", ":")) if term.group_templates else template
        colours.append(("coupling", name, local, rank, len(args), identity))
    groups = _time_groups(term.equal_time_aliases)
    members = {point: group for group in groups for point in group}
    for group in groups:
        colours.append(("equal_time",))
        instances.append([node[point] for point in group])

    # Keep the two roles of each fixed lag distinct, including when an
    # equal-time class can choose a different point as representative.
    lag_arcs = []
    aliases = {point: min(group) for point, group in members.items()}
    for delayed, anchor, lag in normalize_lags(term.time_lags, aliases, integrated):
        relation = len(colours)
        colours.append(("time_lag", float(lag).hex()))
        instances.append([])
        for endpoint, role in ((delayed, 0), (anchor, 1)):
            for point in members.get(endpoint, (endpoint,)):
                lag_arcs.append((relation, node[point], role))

    ranked = {colour: i for i, colour in enumerate(sorted(set(colours)))}
    n_r = sum(prop.kind == "R" for prop in term.propagators)
    graph = PointGraph(labels, [(ranked[c], 0, 0) for c in colours], instances,
                       n_r, len(term.propagators) - n_r)
    adj = graph.adjacency(term.propagators)

    # Keep PointGraph's arc fields for propagators/ownership and append
    # bits for slot-point incidence and the directed absorbed-R flag.
    shift = 5 * graph.bits
    for source, dest in slots:
        adj[source][dest] = adj[source].get(dest, 0) | (1 << shift)
        adj[dest][source] = adj[dest].get(source, 0) | (1 << (shift + 1))
    for left, right in term.r_absorbed_pairs:
        source, dest = node[left], node[right]
        adj[source][dest] = adj[source].get(dest, 0) | (1 << (shift + 2))
        adj[dest][source] = adj[dest].get(source, 0) | (1 << (shift + 3))
    for relation, endpoint, role in lag_arcs:
        adj[relation][endpoint] = 1 << (shift + 4 + 2 * role)
        adj[endpoint][relation] = 1 << (shift + 5 + 2 * role)
    search = Search(adj, shift + (8 if lag_arcs else 4))
    search.descend(list(graph.col0), graph.k0, [])

    payload = (_ID_SCHEMA, sorted(colours), shift, search.best_cert)
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return f"{_ID_SCHEMA}:{hashlib.sha256(encoded).hexdigest()}"


def diagram_ids(term: DiagramTerm) -> tuple[str, ...]:
    """Sorted, unique v1 typed spatial graph IDs represented by ``term``.

    A legacy collected term can represent several typed couplings on the
    same spatial graph. Each ID keeps the named externals, directed R and
    undirected C multiplicities, coupling names, local/non-local layout and
    ordered slots, tensor ranks, equal-time groups and absorbed R edges.
    For the same source template information, internal point names,
    component routing, coefficients and evaluation inputs do not affect the
    IDs. Terms from ``routing="multiplicity"`` also carry the full source
    template identity, including ordered field types. Use this route in
    both engines to compare complete typed orbits.
    """
    source = tuple(
        (template[0], template[1], len(template[4]), points, template)
        for points, template in term.vertex_types
    )
    if term.group_templates:
        # Legacy collection can omit the full vertex type. A grouped
        # occurrence nevertheless retains its ordered field/group incidence.
        annotations = {(name, args): (roles, groups, mode)
                       for name, args, roles, groups, mode in term.group_templates}
        matched = {(name, args) for name, _, _, args, _ in source}
        source = tuple(
            (name, local, rank, args, template + (annotations[name, args],))
            if (name, args) in annotations else (name, local, rank, args, template)
            for name, local, rank, args, template in source
        ) + tuple(
            (name, False, len(roles), args, (roles, groups, mode))
            for name, args, roles, groups, mode in term.group_templates
            if (name, args) not in matched
        )
    represented = {(name, frozenset(args)) for name, _, _, args, _ in source}
    layouts = _coupling_layouts(term.coupling_sum, represented)
    return tuple(sorted({_typed_graph_id(term, source + layout) for layout in layouts}))
