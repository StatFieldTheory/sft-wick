"""The point multigraph of a layout and the individualisation-refinement
search of the ``"builtin"`` method of :mod:`sft_wick._canonical`.

The module needs only the standard library and imports nothing of the
package: :class:`PointGraph` takes the plain data of a
:class:`~sft_wick._canonical.SequenceLayout`.
"""

from __future__ import annotations

import struct
from collections.abc import Hashable, Iterable, Sequence
from typing import NamedTuple, Protocol


class PropagatorLike(Protocol):
    """What a canonical key reads of a propagator."""

    @property
    def kind(self) -> str: ...

    @property
    def spatial_left(self) -> Hashable: ...

    @property
    def spatial_right(self) -> Hashable: ...


class PointGraph:
    """The point multigraph of a layout.

    Every ordered node pair carries one integer label, five fields of
    ``bits`` bits each: R out, R in, C, instance to leg, leg to instance.
    A count of parallel propagators never reaches ``2**bits``, so a label
    records the multiplicity and parallel propagators are one arc.
    """

    __slots__ = ("node", "col0", "k0", "bits", "base", "n_r", "n_c", "entry_bytes")

    def __init__(
        self,
        labels: Sequence[Hashable],
        colours: Sequence[tuple[int, int, int]],
        instance_legs: Sequence[Sequence[int]],
        n_r: int,
        n_c: int,
    ) -> None:
        self.node = {label: i for i, label in enumerate(labels)}
        rank = {colour: i for i, colour in enumerate(sorted(set(colours)))}
        self.col0 = tuple(rank[colour] for colour in colours)
        self.k0 = len(rank)
        self.n_r, self.n_c = n_r, n_c
        bits = max(1, (n_r + max(n_c, 0)).bit_length())
        self.bits = bits
        base: list[dict[int, int]] = [{} for _ in colours]
        first_instance = len(labels)
        for i, legs in enumerate(instance_legs):
            for leg in legs:
                base[first_instance + i][leg] = 1 << 3 * bits
                base[leg][first_instance + i] = 1 << 4 * bits
        self.base = tuple(tuple(row.items()) for row in base)
        n = len(colours)
        self.entry_bytes = (max(n * n - 1, 0).bit_length() + 5 * bits + 7) // 8

    def adjacency(self, props: Iterable[PropagatorLike]) -> list[dict[int, int]]:
        """The labelled arcs of ``props`` and of the instance nodes.

        Raises:
            ValueError: If the sequence admits no complete pairing, a
                propagator has an unknown endpoint label or a kind other
                than ``"R"`` and ``"C"``, or the counts differ from the
                layout's.
        """
        if self.n_c < 0:
            raise ValueError(
                "this vertex sequence admits no complete pairing, so it has no topologies"
            )
        node = self.node
        bits = self.bits
        r_in, c_unit = 1 << bits, 1 << 2 * bits
        adj = [dict(row) for row in self.base]
        n_r = n_c = 0
        for p in props:
            try:
                a = node[p.spatial_left]
                b = node[p.spatial_right]
            except KeyError:
                raise ValueError(
                    f"propagator {p!r} has an endpoint that is neither an "
                    "external nor a point of this vertex sequence"
                ) from None
            kind = p.kind
            if kind == "R":
                row = adj[a]
                row[b] = row.get(b, 0) + 1
                row = adj[b]
                row[a] = row.get(a, 0) + r_in
                n_r += 1
            elif kind == "C":
                row = adj[a]
                row[b] = row.get(b, 0) + c_unit
                if a != b:
                    row = adj[b]
                    row[a] = row.get(a, 0) + c_unit
                n_c += 1
            else:
                raise ValueError(f"propagator {p!r} has kind {kind!r}, not R or C")
        if n_r != self.n_r or n_c != self.n_c:
            raise ValueError(
                f"the topology has {n_r} R and {n_c} C propagators; every "
                f"topology of this sequence has {self.n_r} and {self.n_c}"
            )
        return adj

    def search(self, props: Iterable[PropagatorLike]) -> Search:
        search = Search(self.adjacency(props), 5 * self.bits)
        search.descend(list(self.col0), self.k0, [])
        return search

    def key(self, search: Search, prefix: bytes) -> bytes:
        """``prefix | u32 bits | u32 m | m entries``, each entry the
        unsigned big-endian value of one arc of the best leaf."""
        cert = search.best_cert
        width = self.entry_bytes
        head = struct.pack(">II", self.bits, len(cert))
        return prefix + head + b"".join([e.to_bytes(width, "big") for e in cert])


def _refine(
    nbrs: Sequence[tuple[tuple[int, int], ...]],
    loops: Sequence[int],
    col: list[int],
    k: int,
    shift: int,
) -> tuple[list[int], int]:
    """Colour refinement of the ordered partition ``col`` with ``k`` cells.

    The signature of a node is its colour, the sorted labels and colours
    of its out-arcs, and ``loops``, the label of its arc to itself (0
    without one).  An arc to itself enters the sorted list as an arc to
    another node of the same colour would, so without the last part a
    node with a self-loop and a node of a 2-cycle with the same label
    would share a cell for good.  New colours are the ranks of the sorted
    distinct signatures, so the cell order is kept.  Every arc has a
    reverse arc, so out-arcs are enough.  ``shift`` keeps a colour below
    ``2**shift``.
    """
    while True:
        sigs = [
            (c, tuple(sorted([(lab << shift) | col[w] for w, lab in row])), loop)
            for c, row, loop in zip(col, nbrs, loops, strict=True)
        ]
        ordered = sorted(set(sigs))
        rank = {sig: i for i, sig in enumerate(ordered)}
        col = [rank[sig] for sig in sigs]
        if len(ordered) == k:
            return col, k
        k = len(ordered)


def _orbits(gens: list[list[int]], fixed: list[int], n: int) -> list[int]:
    """The smallest node of each node's orbit under the generators that fix
    every node of ``fixed``."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for gen in gens:
        if all(gen[v] == v for v in fixed):
            for v, image in enumerate(gen):
                a, b = find(v), find(image)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    return [find(v) for v in range(n)]


def _perm(source: list[int], target: list[int]) -> list[int]:
    """The node map that sends the node at each position of the leaf
    ``source`` to the node at that position of the leaf ``target``."""
    at = [0] * len(target)
    for v, position in enumerate(target):
        at[position] = v
    return [at[position] for position in source]


def _common_prefix(a: list[int], b: list[int]) -> int:
    depth = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        depth += 1
    return depth


class _Leaf(NamedTuple):
    """A leaf of the search: its discrete colouring (the position of each
    node), its certificate and the nodes individualised on its path."""

    col: list[int]
    cert: tuple[int, ...]
    path: list[int]


class Search:
    """Individualisation-refinement search for the smallest leaf
    certificate, depth first.

    A child is skipped when a known automorphism that fixes the current
    path maps it into the orbit of an explored sibling.  A leaf equal to the
    first or the best leaf gives an automorphism and a jump back to the
    level where the two paths part.  Both only drop subtrees that are images
    of explored ones, so the best certificate is the smallest over the whole
    tree, whatever the node numbering.  The group order is the product, over
    the levels of the first path, of the orbit size of the first path's node
    inside its target cell.

    Unlike nauty, the search does not prune by node invariants.  Where
    refinement cannot split the orbits of a target cell, it visits every
    leaf that no known automorphism maps onto an explored one, and their
    number can grow exponentially with the graph: disjoint directed cycles
    of two lengths and Cai-Fürer-Immerman graphs are such cases.
    ``descend`` recurses once per level, so a cell of about 1,000 nodes
    that refinement never splits exceeds Python's default recursion limit
    and raises :class:`RecursionError`.  Topologies of the engine are far
    smaller.
    """

    __slots__ = ("nbrs", "loops", "edges", "n", "shift", "cert_shift", "gens",
                 "first_cells", "first", "best")

    def __init__(self, adj: list[dict[int, int]], cert_shift: int) -> None:
        n = len(adj)
        self.n = n
        self.nbrs = [tuple(row.items()) for row in adj]
        #: The label of each node's arc to itself, 0 without one.
        self.loops = [row.get(v, 0) for v, row in enumerate(adj)]
        self.edges = [(u, w, lab) for u, row in enumerate(self.nbrs) for w, lab in row]
        self.shift = (2 * n).bit_length()
        self.cert_shift = cert_shift
        self.gens: list[list[int]] = []
        #: The target cell of each level of the first path.
        self.first_cells: list[list[int]] = []
        self.first: _Leaf | None = None
        self.best: _Leaf | None = None

    def descend(self, col: list[int], k: int, path: list[int]) -> int | None:
        """Search below the node ``path`` with partition ``col``.  Returns
        the depth to jump back to, or ``None``."""
        col, k = _refine(self.nbrs, self.loops, col, k, self.shift)
        n = self.n
        if k == n:
            return self._leaf(col, path)
        sizes = [0] * k
        for c in col:
            sizes[c] += 1
        target = next(c for c, size in enumerate(sizes) if size > 1)
        cell = [v for v, c in enumerate(col) if c == target]
        if self.first is None:
            self.first_cells.append(cell)
        doubled = [2 * c for c in col]
        for v in cell:
            doubled[v] += 1
        depth = len(path)
        explored: list[int] = []
        orbit_of: list[int] = []
        known = 0
        for v in cell:
            if explored and self.gens:
                if known != len(self.gens):
                    orbit_of = _orbits(self.gens, path, n)
                    known = len(self.gens)
                root = orbit_of[v]
                if any(orbit_of[u] == root for u in explored):
                    continue
            child = doubled[:]
            child[v] -= 1
            jump = self.descend(child, k + 1, path + [v])
            explored.append(v)
            if jump is not None and jump < depth:
                return jump
        return None

    def _leaf(self, col: list[int], path: list[int]) -> int | None:
        """Compare a leaf with the first and the best leaf.  Returns the
        depth to jump back to, or ``None``."""
        n, shift = self.n, self.cert_shift
        cert = tuple(
            sorted([((col[u] * n + col[w]) << shift) | lab for u, w, lab in self.edges])
        )
        leaf = _Leaf(col, cert, path)
        if self.first is None:
            self.first = self.best = leaf
            return None
        for known in (self.first, self.best):
            if cert == known.cert:
                self.gens.append(_perm(known.col, col))
                return _common_prefix(path, known.path)
        if cert < self.best.cert:
            self.best = leaf
        return None

    @property
    def best_cert(self) -> tuple[int, ...]:
        return self.best.cert

    def group_order(self) -> int:
        order = 1
        path = self.first.path
        for level, cell in enumerate(self.first_cells):
            orbit_of = _orbits(self.gens, path[:level], self.n)
            root = orbit_of[path[level]]
            order *= sum(1 for v in cell if orbit_of[v] == root)
        return order
