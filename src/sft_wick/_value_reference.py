"""Independent numeric reference for a small labelled Wick contraction.

Inputs specify individual operators, vertices and numeric tensors. No
contraction-engine types, topology keys or routing records enter this module.
"""

from __future__ import annotations

from functools import lru_cache
from math import fsum, prod
from typing import NamedTuple

import numpy as np


class Slot(NamedTuple):
    """An operator's point, owner, response status and component coordinate."""

    point: int
    owner: int
    response: bool
    axis: int | None
    fixed: int | None
    width: int
    dimension: int
    absorbed: bool = False
    clock: int | None = None

    @property
    def time_point(self):
        return self.point if self.clock is None else self.clock


class ValueLayoutError(ValueError):
    """The numeric fixture cannot realize a layout's contractions."""


def _matchings(slots, remaining):
    """Enumerate pairings of operator occurrences, with local pruning only."""
    if not remaining:
        yield ()
        return
    a, *tail = remaining
    for j, b in enumerate(tail):
        left, right = slots[a], slots[b]
        if left.response and right.response:
            continue
        if left.response != right.response and left.time_point == right.time_point:
            continue
        for suffix in _matchings(slots, tail[:j] + tail[j + 1:]):
            yield ((a, b), *suffix)


def _acyclic(edges):
    successors = {}
    for a, b in edges:
        successors.setdefault(a, set()).add(b)
    # Repeatedly remove vertices with no outgoing edge.
    while successors:
        sinks = set().union(*successors.values()) - successors.keys()
        if not sinks:
            return False
        successors = {a: rest for a, bs in successors.items() if (rest := bs - sinks)}
    return True


def _connected(pairs, slots, n_owners):
    neighbors = [set() for _ in range(n_owners)]
    for i, j in pairs:
        a, b = slots[i].owner, slots[j].owner
        neighbors[a].add(b)
        neighbors[b].add(a)
    seen = {0}
    todo = [0]
    while todo:
        new = neighbors[todo.pop()] - seen
        seen.update(new)
        todo.extend(new)
    return len(seen) == n_owners


def numerical_sum(values):
    """Accumulate the two parts independently, including for pure R phases."""
    values = list(values)
    return complex(fsum(x.real for x in values), fsum(x.imag for x in values))


@lru_cache(maxsize=1024)
def _einsum_path(shapes, axes):
    # Broadcast dummy data has no allocated tensor storage. Path selection
    # depends only on the index incidence and axis sizes.
    args = []
    for shape, indices in zip(shapes, axes):
        args.extend((np.broadcast_to(0, shape), indices))
    return tuple(np.einsum_path(*args, [], optimize="greedy")[0])


def contract(factors):
    """Contraction of array factors with integer axis labels."""
    coefficient = 1
    arrays, axes = [], []
    renumber = {}
    for array, indices in factors:
        if not indices:
            coefficient *= array.item()
        else:
            arrays.append(array)
            axes.append(tuple(renumber.setdefault(i, len(renumber)) for i in indices))
    if not arrays or coefficient == 0:
        return coefficient
    args = []
    for array, indices in zip(arrays, axes):
        args.extend((array, indices))
    shapes = tuple(array.shape for array in arrays)
    dimensions = {i: n for ids, shape in zip(axes, shapes) for i, n in zip(ids, shape)}
    path = False
    if prod(dimensions.values()) > 128:
        path = list(_einsum_path(shapes, tuple(axes)))
    return coefficient * np.einsum(*args, [], optimize=path).item()


def reference_value(slots, couplings, n_owners, connected, prefactor, matrix, diagonal):
    """Sum labelled pairings with a direct definition of the MSR phase.

    ``matrix(kind, left, right)`` provides the shared numeric fixture.
    ``couplings`` lists independent tensors and their operator-axis labels.
    """
    if len(slots) % 2 or connected and len(slots) // 2 < n_owners - 1:
        return 0j
    n_response = sum(s.response for s in slots)

    def values():
        for pairs in _matchings(slots, list(range(len(slots)))):
            response_edges = []
            propagators = []
            for i, j in pairs:
                a, b = slots[i], slots[j]
                if a.response:
                    a, b = b, a
                if b.response:
                    response_edges.append((a.time_point, b.time_point))
                propagators.append(("R" if b.response else "C", a, b))
            if not _acyclic(response_edges):
                continue
            if connected and not _connected(pairs, slots, n_owners):
                continue
            factors = list(couplings)
            for kind, a, b in propagators:
                absorbed = kind == "R" and b.absorbed
                if (absorbed or diagonal[kind]) and a.dimension != b.dimension and (
                    a.axis is not None or b.axis is not None
                ):
                    raise ValueLayoutError("an index reduction joins unequal component dimensions")
                if (a.dimension == 1) != (b.dimension == 1):
                    raise ValueLayoutError("a propagator joins scalar and vector fields")
                values = matrix(kind, a.point, b.point, absorbed)
                sub = (a.fixed if a.axis is None else slice(a.width),
                       b.fixed if b.axis is None else slice(b.width))
                axes = tuple(s.axis for s in (a, b) if s.axis is not None)
                factors.append((np.asarray(values[sub]), axes))
            yield contract(factors)

    return prefactor * (-1j) ** n_response * numerical_sum(values())
