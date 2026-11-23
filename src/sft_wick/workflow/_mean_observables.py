"""Reconstruct observables of a shifted field from centered Wick diagrams."""

from __future__ import annotations

from dataclasses import replace
from itertools import chain, combinations

from ..expressions import Propagator, Rational, Symbol
from ..fields import Field
from ..indices import fresh_names, observable_labels
from ..perturbation import (
    DiagramInfo,
    DiagramTerm,
    PerturbativeResult,
    _diagram_expression,
    _expression_sum,
    compute_moment,
)
from ..simplify import _apply_index_sub
from ..vertices import Vertex, VertexInstance


def _removed_slots(ops, target):
    yield ()
    if target == "connected" and len(ops) > 1:
        return
    physical = tuple(i for i, op in enumerate(ops) if op.is_physical)
    for count in range(1, len(physical) + 1):
        yield from combinations(physical, count)


def observable_subsets(ops, target):
    """Yield (kept, removed) operator tuples for an additive mean shift."""
    ops = tuple(ops)
    for removed in _removed_slots(ops, target):
        yield (
            tuple(op for i, op in enumerate(ops) if i not in removed),
            tuple(ops[i] for i in removed),
        )


def _rename_collisions(labels, spatial, components):
    """Keep restored external labels distinct from partial diagram dummies."""
    spatial = set(spatial)
    components = set(components)
    taken = labels | spatial | components
    points = sorted(labels & spatial)
    point_names = fresh_names("y", len(points), taken)
    indices = sorted(labels & components)
    index_names = fresh_names("i", len(indices), taken | set(point_names))
    return dict(zip(points, point_names)), dict(zip(indices, index_names))


def _extend_term(term, labels, removed, mean_symbol):
    components = tuple(index for index, _ in term.summation_indices)
    points, indices = _rename_collisions(labels, term.integration_vars, components)
    taken = labels | set(term.integration_vars) | set(components)
    taken |= set(points.values()) | set(indices.values())
    legs = fresh_names("y", len(removed), taken)

    coupling = _apply_index_sub(term.coupling_sum, {**points, **indices})
    props = [
        replace(
            p,
            spatial_left=points.get(p.spatial_left, p.spatial_left),
            spatial_right=points.get(p.spatial_right, p.spatial_right),
            index_left=indices.get(p.index_left, p.index_left),
            index_right=indices.get(p.index_right, p.index_right),
        )
        for p in term.propagators
    ]
    # An absorbed R carries the point through the usual domain and
    # direction rules. Its numeric factor is one; the mean coupling
    # includes +i to cancel the diagram's extra response phase of -i.
    for op, leg in zip(removed, legs):
        index = None if op.field.is_scalar else op.component_index
        props.append(Propagator("R", index, index, op.spatial_arg, leg))
        coupling = coupling * Symbol(mean_symbol, () if index is None else (index,), (leg,))

    pairs = tuple((op.spatial_arg, leg) for op, leg in zip(removed, legs))
    return replace(
        term,
        propagators=tuple(props),
        coupling_sum=coupling,
        integration_vars=tuple(points.get(v, v) for v in term.integration_vars) + tuple(legs),
        summation_indices=tuple((indices.get(i, i), n) for i, n in term.summation_indices),
        n_response=term.n_response + len(legs),
        vertex_types=tuple(
            (tuple(points.get(point, point) for point in labels), template)
            for labels, template in term.vertex_types
        ),
        equal_time_aliases=tuple(
            (points.get(left, left), points.get(right, right))
            for left, right in term.equal_time_aliases
        ) + tuple((leg, point) for point, leg in pairs),
        group_templates=tuple(
            (name, tuple(points.get(p, p) for p in coords), roles, groups, mode)
            for name, coords, roles, groups, mode in term.group_templates
        ),
        time_lags=tuple(
            (points.get(d, d), points.get(a, a), lag) for d, a, lag in term.time_lags
        ),
        r_absorbed_pairs=tuple(
            (points.get(left, left), points.get(right, right))
            for left, right in term.r_absorbed_pairs
        ) + pairs,
    )


def _extend_info(info, observable, kept, removed, labels, mean_symbol):
    instances = info.vertex_instances
    spatial = [p for iv in instances for p in iv.spatial_variables]
    components = [i for iv in instances for i in iv.component_indices]
    points, indices = _rename_collisions(labels, spatial, components)
    if points or indices:
        names = {**points, **indices}
        instances = [
            replace(
                iv,
                field_operators=[
                    replace(
                        op,
                        spatial_arg=points.get(op.spatial_arg, op.spatial_arg),
                        component_index=indices.get(op.component_index, op.component_index),
                    )
                    for op in iv.field_operators
                ],
                coupling_symbol=_apply_index_sub(iv.coupling_symbol, names),
                spatial_variables=[points.get(p, p) for p in iv.spatial_variables],
                component_indices=[indices.get(i, i) for i in iv.component_indices],
                time_lags=tuple(
                    (points.get(d, d), points.get(a, a), lag) for d, a, lag in iv.time_lags
                ),
                equal_time_aliases=tuple(
                    (points.get(left, left), points.get(right, right))
                    for left, right in iv.equal_time_aliases
                ),
            )
            for iv in instances
        ]
    taken = labels | set(spatial) | set(components) | set(points.values()) | set(indices.values())
    legs = fresh_names("y", len(removed), taken)
    operators = sum(len(iv.field_operators) for iv in instances)
    old_to_new = kept + tuple(range(len(observable), len(observable) + operators))
    pairing = [(old_to_new[a], old_to_new[b]) for a, b in info.pairing]
    means = []
    for slot, leg in zip(removed, legs):
        op = observable[slot]
        psi = Field("psi", "response", op.field.n_components)
        index = None if op.field.is_scalar else op.component_index
        indices_at_leg = [] if index is None else [index]
        partner = psi(leg) if index is None else psi(index, leg)
        vertex = Vertex((psi,), mean_symbol, local=False, already_R_contracted=True)
        coupling = Symbol(mean_symbol, tuple(indices_at_leg), (leg,))
        means.append(VertexInstance(
            vertex, [partner], coupling, [leg], indices_at_leg, len(instances) + len(means)
        ))
        pairing.append((slot, len(observable) + operators + len(means) - 1))
    return DiagramInfo(list(observable), [*instances, *means], tuple(pairing),
                       info.coefficient, info.order)


def expand_with_mean(observable, action, max_order, *, mean_symbol, **compute_kwargs):
    """Build the Wick expansion of the original observable after shifting the action.

    Each deterministic external factor multiplies a partial fluctuation diagram
    without using up a perturbative order. The caller supplies its coupling
    as a callable returning i times the propagated initial mean.
    """
    observable = tuple(observable)
    target = compute_kwargs.get("target", "moment")
    slots = iter(_removed_slots(observable, target))
    next(slots)
    first = next(slots, None)
    if first is not None:
        if compute_kwargs.get("certify", False):
            raise ValueError("certify=True cannot certify reconstructed mean factors")
        if max_order and not compute_kwargs.get("collect_topology", True):
            raise ValueError("reconstructed mean factors require collect_topology=True")
    result = compute_moment(list(observable), action, max_order, **compute_kwargs)
    if first is None:
        return result

    terms = {order: list(result.diagram_terms(order)) for order in range(max_order + 1)}
    infos = {order: list(result.diagrams_by_order[order]) for order in terms}
    labels = observable_labels(observable)
    for omitted in chain((first,), slots):
        kept = tuple(i for i in range(len(observable)) if i not in omitted)
        removed = tuple(observable[i] for i in omitted)
        if kept:
            partial = compute_moment(
                [observable[i] for i in kept], action, max_order, **compute_kwargs
            )
            partial_terms = partial.diagram_terms_by_order
            partial_infos = partial.diagrams_by_order
        else:
            response_phase = compute_kwargs.get("response_phase", True)
            partial_terms = {
                0: [DiagramTerm((), Rational(1), Rational(1), (), (), 0,
                                response_phase=response_phase)]
            }
            partial_infos = {0: [DiagramInfo([], [], (), Rational(1), 0)]}
        for order, diagrams in partial_terms.items():
            terms[order].extend(
                _extend_term(dt, labels, removed, mean_symbol) for dt in diagrams
            )
            infos[order].extend(
                _extend_info(di, observable, kept, omitted, labels, mean_symbol)
                for di in partial_infos.get(order, ())
            )

    expressions = {
        order: _expression_sum(_diagram_expression(dt) for dt in diagrams)
        for order, diagrams in terms.items()
    }
    return PerturbativeResult(expressions, _expression_sum(expressions.values()),
                              infos, terms, target=target)
