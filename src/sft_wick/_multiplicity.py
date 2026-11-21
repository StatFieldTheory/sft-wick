"""Weighted spatial topologies for leg-symmetric coupling tensors."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from itertools import permutations, product
from math import factorial, prod

from . import wick
from ._canonical import Keyer, SequenceLayout
from ._connected import ConnectedIncidence
from .expressions import Propagator
from .fields import FieldOperator
from .wick import Pairing


WeightedTopology = tuple[list[Propagator], Pairing, int, int]

# Bound explicit group storage. At order six this includes every layout.
_MAX_EXPLICIT_GROUP_ORDER = factorial(6)


def weighted_topologies(
    operators: Sequence[FieldOperator],
    layout: SequenceLayout,
    keyer: Keyer,
    incidence: ConnectedIncidence | None,
) -> Iterator[WeightedTopology]:
    """Yield ``(props, pairing, routings per topology, labelled copies)``.

    Representatives are first in Wick topology enumeration order. Vertex
    types and external points are fixed by the relabelling group. The caller
    must ensure that the tensor factors are equivariant between labelled
    copies and symmetric in the local legs that share a point and field type.
    """
    if not operators:
        return
    if incidence is not None and not incidence.disjoint_labels:
        raise ValueError("weighted topologies require disjoint occurrence labels")

    phi_at, psi_at = wick._operators_at(operators)
    slot_choices = prod(
        factorial(len(slots)) for slots in (*phi_at.values(), *psi_at.values())
    )
    if layout.group_order <= _MAX_EXPLICIT_GROUP_ORDER:
        yield from _weighted_orbits(
            operators, layout, incidence, phi_at, psi_at, slot_choices
        )
    else:
        yield from _weighted_by_key(
            operators, keyer, incidence, phi_at, psi_at, slot_choices
        )


def _routing_count(slot_choices, r_edges, c_edges) -> int:
    """Count the orbit of one pairing under local leg permutations.

    A stabiliser may exchange parallel edges of one kind. It can also
    reverse each C self-loop. An R edge is directed and has no such flip.
    """
    r_counts = Counter(r_edges)
    c_counts = Counter(c_edges)
    parallel_edges = prod(factorial(n) for n in (*r_counts.values(), *c_counts.values()))
    loop_flips = 1 << sum(n for (a, b), n in c_counts.items() if a == b)
    count, remainder = divmod(slot_choices, parallel_edges * loop_flips)
    if remainder:
        raise RuntimeError("the stabiliser of a pairing does not divide its leg group")
    return count


def _point_permutations(layout: SequenceLayout) -> tuple[tuple[int, ...], ...]:
    """The layout's point permutations, with non-local instances intact."""
    n_points = len(layout.labels)
    attached = {leg for legs in layout.instance_legs for leg in legs}
    cells: dict[tuple[int, int, int], list[tuple[int, ...]]] = {}
    for point, colour in enumerate(layout.colours[:n_points]):
        if point not in attached:
            # External points have unique colours and their cells are singletons.
            cells.setdefault(colour, []).append((point,))
    for colour, legs in zip(layout.colours[n_points:], layout.instance_legs, strict=True):
        cells.setdefault(colour, []).append(legs)

    movable = tuple(cell for cell in cells.values() if len(cell) > 1)
    source = tuple(point for cell in movable for legs in cell for point in legs)
    maps = []
    for cell_images in product(*(permutations(cell) for cell in movable)):
        image = (point for cell in cell_images for legs in cell for point in legs)
        permutation = list(range(n_points))
        for a, b in zip(source, image, strict=True):
            permutation[a] = b
        maps.append(tuple(permutation))
    if len(maps) != layout.group_order:
        raise RuntimeError("point permutations do not match the layout group order")
    return tuple(maps)


def _r_image(edges, permutation):
    return tuple(sorted((permutation[a], permutation[b]) for a, b in edges))


def _c_image(edges, permutation):
    ends = ((permutation[a], permutation[b]) for a, b in edges)
    return tuple(sorted((a, b) if a <= b else (b, a) for a, b in ends))


def _weighted_orbits(operators, layout, incidence, phi_at, psi_at, slot_choices):
    # Quotient directed R assignments first. Every relabelling that carries
    # one R assignment to another also carries its leftover phi capacities;
    # the C completions of an R representative need only be quotiented by
    # the stabiliser of that R assignment.
    node = {point: i for i, point in enumerate(layout.labels)}
    permutations = _point_permutations(layout)
    identity = tuple(range(len(node)))
    seen_r = set()
    for r_edges, remaining in wick._r_classes_with_leftovers(phi_at, psi_at):
        ranked_r = tuple((node[a], node[b]) for a, b in r_edges)
        identity_r = _r_image(ranked_r, identity)
        if identity_r in seen_r:
            continue

        stabiliser = []
        for permutation in permutations:
            image_r = _r_image(ranked_r, permutation)
            seen_r.add(image_r)
            if image_r == identity_r:
                stabiliser.append(permutation)
        r_copies = len(permutations) // len(stabiliser)

        seen_c = set()
        for c_edges in wick._enumerate_c_pairings(remaining):
            ranked_c = tuple((node[a], node[b]) for a, b in c_edges)
            identity_c = _c_image(ranked_c, identity)
            if identity_c in seen_c:
                continue
            images_c = {_c_image(ranked_c, permutation) for permutation in stabiliser}
            seen_c.update(images_c)

            props, pairing = wick._representative(operators, phi_at, psi_at, r_edges, c_edges)
            if incidence is None or incidence.is_connected(pairing):
                count = _routing_count(slot_choices, r_edges, c_edges)
                yield props, pairing, count, r_copies * len(images_c)


def _weighted_by_key(operators, keyer, incidence, phi_at, psi_at, slot_choices):
    # For a larger relabelling group, stream labelled topologies through
    # the existing canonicaliser and store only the first member per class.
    classes: dict[bytes, WeightedTopology] = {}
    for r_edges, remaining in wick._r_classes_with_leftovers(phi_at, psi_at):
        seen_c = set()
        for c_edges in wick._enumerate_c_pairings(remaining):
            signature = tuple(sorted(c_edges))
            if signature in seen_c:
                continue
            seen_c.add(signature)

            props, pairing = wick._representative(operators, phi_at, psi_at, r_edges, c_edges)
            if incidence is not None and not incidence.is_connected(pairing):
                continue
            key = keyer.key(props)
            representative = classes.get(key)
            if representative is None:
                classes[key] = (props, pairing, _routing_count(slot_choices, r_edges, c_edges), 1)
            else:
                first_props, first_pairing, count, copies = representative
                classes[key] = (first_props, first_pairing, count, copies + 1)
    yield from classes.values()
