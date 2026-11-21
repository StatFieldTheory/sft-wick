"""Vertex models, hand-built graphs and plain-data conversions shared by the
canonical-key tests (``tests/test_typed_canonical_key.py``).

An engine case is one vertex sequence of a model with the topologies that
:func:`sft_wick.wick.wick_contract_spatial` returns for it.  A hand-built
case is a list of topologies on a layout given as plain data, for graphs the
engine does not produce: stars, trees, regular graphs that colour
refinement cannot split, and parallel propagators.  Both carry a
:class:`PlainLayout`, the arguments of ``SequenceLayout.from_plain``, which
:func:`brute_structure` turns into the plain data of :mod:`tests.brute_relabel`.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from math import factorial, prod

from sft_wick import Action, Field, Vertex
from sft_wick.expressions import Product, Propagator, Symbol
from sft_wick.perturbation import _vertex_sequences
from sft_wick.wick import (
    _enumerate_c_pairings,
    _operators_at,
    _r_classes_with_leftovers,
    wick_contract_spatial,
)

from . import brute_relabel as brute

PHI = Field("phi", "physical")
PSI = Field("psi", "response")

#: Local vertices.  F and G have one shape and differ only in name.
F = Vertex([PSI, PHI, PHI], "F")
G = Vertex([PSI, PHI, PHI], "G")
Q = Vertex([PSI, PHI, PHI, PHI], "Q")
M = Vertex([PSI, PSI, PHI, PHI], "M")
P5 = Vertex([PSI] + [PHI] * 5, "P5")

#: Non-local vertices.  K and K_EQ share a name and differ in a flag.
K = Vertex([PSI] * 3, "K", local=False)
K_EQ = Vertex([PSI] * 3, "K", local=False, equal_time=True)
K_R = Vertex([PSI] * 3, "Kr", local=False, already_R_contracted=True)
K2 = Vertex([PSI] * 2, "K2", local=False)
L = Vertex([PHI, PSI], "L", local=False)

#: Same names and shapes as F and K, other values: F with a two-component
#: phi, and K with the R flag.
PHI2 = Field("phi", "physical", n_components=2)
F_N2 = Vertex([PSI, PHI2, PHI2], "F")
K_RK = Vertex([PSI] * 3, "K", local=False, already_R_contracted=True)

XY = (("phi", "x"), ("phi", "y"))
XYZ = XY + (("phi", "z"),)
WXYZ = (("phi", "w"),) + XYZ
SIX = tuple(("phi", f"x{k}") for k in range(1, 7))
PHI_X_PSI_Y = (("phi", "x"), ("psi", "y"))

#: name -> (vertex templates of the action, observable)
MODELS = {
    "cubic": ((F,), XY),
    "cubic_xyz": ((F,), XYZ),
    "cubic_response": ((F,), PHI_X_PSI_Y),
    "F_G": ((F, G), XY),
    "quartic": ((Q,), XY),
    "M": ((M,), XY),
    "psi_phi5": ((P5,), WXYZ),
    "F_K": ((F, K), XYZ),
    "F_Kr": ((F, K_R), XYZ),
    "F_L": ((F, L), XY),
    "F_K2": ((F, K2), XY),
    "K_Keq": ((K, K_EQ), SIX),
    "F_Fn2": ((F, F_N2), XY),
    "K_Kr": ((K, K_RK), SIX),
}


def observable(spec: tuple[tuple[str, str], ...]) -> list:
    """Fresh field operators for ``(field, label)`` pairs."""
    field = {"phi": PHI, "psi": PSI}
    return [field[kind](label) for kind, label in spec]


@dataclass(frozen=True)
class PlainLayout:
    """The arguments of ``SequenceLayout.from_plain``."""

    externals: tuple[str, ...]
    local_points: tuple[tuple[str, Vertex], ...]
    nonlocal_instances: tuple[tuple[Vertex, tuple[str, ...]], ...]
    n_r: int
    n_c: int

    def kwargs(self) -> dict:
        return {
            "externals": self.externals,
            "local_points": self.local_points,
            "nonlocal_instances": self.nonlocal_instances,
            "n_r": self.n_r,
            "n_c": self.n_c,
        }


@dataclass(frozen=True, eq=False)
class Case:
    """Topologies on one layout, each a list of propagators.  ``seq`` and
    ``observable`` are set for an engine case and are ``None`` for a
    hand-built one.  ``closed`` says whether the topologies are every
    labelled topology of the sequence, so that each class holds a whole
    orbit.  Cases compare and hash by identity, so cached ones can key a
    cache."""

    label: str
    plain: PlainLayout
    topologies: tuple[list[Propagator], ...]
    seq: object = None
    observable: tuple | None = None
    closed: bool = True


def sequence_label(seq) -> str:
    """The vertices of a sequence, as in ``F,F,K[eq]``."""
    return ",".join(_vertex_label(vi.vertex) for vi in seq.instances)


def _vertex_label(vertex: Vertex) -> str:
    flags = ("[eq]" if vertex.equal_time else "") + (
        "[R]" if vertex.already_R_contracted else ""
    )
    components = sorted({f.n_components for f in vertex.fields} - {1})
    flags += "".join(f"[N{n}]" for n in components)
    return vertex.coupling + flags


def plain_layout(seq, observable_ops) -> PlainLayout:
    """The plain layout of an engine vertex sequence."""
    n_r = sum(1 for op in seq.operators if op.field.is_response)
    n_c, odd = divmod(len(seq.operators) - 2 * n_r, 2)
    assert odd == 0 and n_c >= 0
    return PlainLayout(
        externals=tuple(sorted({op.spatial_arg for op in observable_ops})),
        local_points=tuple(
            (vi.spatial_variables[0], vi.vertex) for vi in seq.instances if vi.vertex.local
        ),
        nonlocal_instances=tuple(
            (vi.vertex, tuple(vi.spatial_variables))
            for vi in seq.instances
            if not vi.vertex.local
        ),
        n_r=n_r,
        n_c=n_c,
    )


@lru_cache(maxsize=None)
def engine_cases(model: str, order: int, select: tuple[Vertex, ...] | None = None):
    """The cases of ``model`` at ``order``, one per vertex sequence with at
    least one topology; with ``select``, only the sequence of those
    vertices (in any order)."""
    vertices, obs_spec = MODELS[model]
    obs = observable(obs_spec)
    cases = []
    for seq in _vertex_sequences(obs, Action(list(vertices)), order, "moment"):
        if select is not None and Counter(vi.vertex for vi in seq.instances) != Counter(select):
            continue
        topologies = tuple(
            list(props) for props, _pairing in wick_contract_spatial(seq.operators).values()
        )
        if topologies:
            cases.append(
                Case(sequence_label(seq), plain_layout(seq, obs), topologies, seq, tuple(obs))
            )
    return tuple(cases)


def engine_case(model: str, order: int, label: str) -> Case:
    """The case of ``model`` at ``order`` whose sequence label is ``label``."""
    (case,) = [c for c in engine_cases(model, order) if c.label == label]
    return case


def stream_topologies(seq):
    """The labelled topologies of ``seq`` one at a time, in the order of
    :func:`sft_wick.wick.wick_contract_spatial`, without holding them all.

    The engine drops a C pairing that comes twice by its signature.  R
    classes are distinct multisets of R edges, so a repeat can only come
    within one R class, and the set of C pairings seen is kept per class.
    """
    phi_at, psi_at = _operators_at(seq.operators)
    for r_assign, remaining in _r_classes_with_leftovers(phi_at, psi_at):
        r_props = [Propagator("R", None, None, a, b) for a, b in r_assign]
        seen: set[tuple[tuple[str, str], ...]] = set()
        for c_pairing in _enumerate_c_pairings(remaining):
            c_edges = tuple(sorted(tuple(sorted(pair)) for pair in c_pairing))
            if c_edges in seen:
                continue
            seen.add(c_edges)
            yield r_props + [Propagator("C", None, None, a, b) for a, b in c_edges]


def plain_topology(props) -> tuple[tuple[str, str, str], ...]:
    return tuple((p.kind, p.spatial_left, p.spatial_right) for p in props)


def brute_structure(plain: PlainLayout):
    """``(points, local_vertex, instances)`` for :mod:`tests.brute_relabel`,
    with a vertex token per distinct vertex value."""
    tokens: list[Vertex] = []

    def token(vertex: Vertex) -> int:
        for i, known in enumerate(tokens):
            if known == vertex:
                return i
        tokens.append(vertex)
        return len(tokens) - 1

    local_vertex = {label: token(vertex) for label, vertex in plain.local_points}
    instances = tuple((token(vertex), legs) for vertex, legs in plain.nonlocal_instances)
    legs = [label for _token, leg_tuple in instances for label in leg_tuple]
    return tuple(sorted(list(local_vertex) + legs)), local_vertex, instances


def partition(keys) -> tuple[int, ...]:
    """The index of the first equal key, for each key: two key lists give
    one partition exactly when these tuples are equal."""
    first: dict = {}
    return tuple(first.setdefault(key, i) for i, key in enumerate(keys))


@lru_cache(maxsize=None)
def brute_orbits(case: Case):
    """The reference group of ``case`` and ``(class, stabiliser)`` of each
    of its topologies."""
    group = brute.relabelling_group(*brute_structure(case.plain))
    data = tuple(brute.orbit_data(plain_topology(p), group) for p in case.topologies)
    return group, data


def twin_factor(props) -> int:
    """The product of m! over the multiplicities m of equal propagators."""
    counts = Counter(brute.image(plain_topology(props), {}))
    return prod(factorial(m) for m in counts.values())


def classes_and_blocks(terms, names: list[str]) -> tuple[int, int]:
    """Reference classes of the engine terms whose local vertices have the
    coupling ``names`` (sorted), from each term's propagators and the points
    of its coupling symbols; and the number of runs of one class in the
    order of the terms."""
    classes = []
    groups: dict = {}
    for term in terms:
        expr = term.coupling_sum
        factors = expr.factors if isinstance(expr, Product) else ()
        assert all(isinstance(f, Symbol) for f in factors)
        if sorted(f.name for f in factors) != names:
            continue
        local_vertex = {f.spatial_args[0]: f.name for f in factors}
        structure = tuple(sorted(local_vertex.items()))
        if structure not in groups:
            groups[structure] = brute.relabelling_group(sorted(local_vertex), local_vertex, ())
        topology = plain_topology(term.propagators)
        classes.append(brute.orbit_data(topology, groups[structure])[0])
    changes = sum(1 for a, b in zip(classes, classes[1:]) if a != b)
    return len(set(classes)), (changes + 1 if classes else 0)


def relabel(props, sigma: dict[str, str]) -> list[Propagator]:
    """``props`` with every label moved by ``sigma``."""
    return [
        Propagator(
            p.kind,
            p.index_left,
            p.index_right,
            sigma.get(p.spatial_left, p.spatial_left),
            sigma.get(p.spatial_right, p.spatial_right),
        )
        for p in props
    ]


# ----------------------------------------------------------------------
# hand-built graphs
# ----------------------------------------------------------------------


def _props(edges) -> list[Propagator]:
    return [Propagator(kind, None, None, a, b) for kind, a, b in edges]


def _local_vertex(name: str, n_psi: int, n_phi: int) -> Vertex:
    return Vertex([PSI] * n_psi + [PHI] * n_phi, name)


def _counts(edges) -> tuple[int, int]:
    n_r = sum(1 for kind, _a, _b in edges if kind == "R")
    return n_r, len(edges) - n_r


def star(k: int) -> Case:
    """``x -R-> r``, ``r -R-> l_i`` and a C loop at each ``l_i``:
    |Aut_V| = k!."""
    edges = [("R", "x", "r")]
    for i in range(k):
        edges += [("R", "r", f"l{i}"), ("C", f"l{i}", f"l{i}")]
    leaf = _local_vertex("Leaf", 1, 2)
    local = (("r", _local_vertex("Root", 1, k)),) + tuple((f"l{i}", leaf) for i in range(k))
    plain = PlainLayout(("x",), local, (), *_counts(edges))
    return Case(f"star k={k}", plain, (_props(edges),), closed=False)


def tree(k: int, j: int) -> Case:
    """``x -R-> r -R-> m_i -R-> l_iq`` with a C loop at each leaf:
    |Aut_V| = k! (j!)^k."""
    edges = [("R", "x", "r")]
    middle = _local_vertex("Mid", 1, j)
    leaf = _local_vertex("Leaf", 1, 2)
    local = [("r", _local_vertex("Root", 1, k))]
    for i in range(k):
        edges.append(("R", "r", f"m{i}"))
        local.append((f"m{i}", middle))
        for q in range(j):
            edges += [("R", f"m{i}", f"l{i}_{q}"), ("C", f"l{i}_{q}", f"l{i}_{q}")]
            local.append((f"l{i}_{q}", leaf))
    plain = PlainLayout(("x",), tuple(local), (), *_counts(edges))
    return Case(f"tree {k}x{j}", plain, (_props(edges),), closed=False)


def _cycle(nodes: list[int]) -> list[tuple[int, int]]:
    return [(nodes[i], nodes[(i + 1) % len(nodes)]) for i in range(len(nodes))]


def _complete(nodes: list[int]) -> list[tuple[int, int]]:
    return [(a, b) for i, a in enumerate(nodes) for b in nodes[i + 1 :]]


def _regular(label: str, n: int, graphs: dict[str, list[tuple[int, int]]]) -> Case:
    """C-only graphs on ``n`` points of one type, all of one degree."""
    degree = 2 * len(next(iter(graphs.values()))) // n
    vertex = _local_vertex(f"Deg{degree}", 0, degree)
    points = tuple(f"p{i}" for i in range(n))
    topologies = tuple(
        _props(("C", f"p{a}", f"p{b}") for a, b in edges) for edges in graphs.values()
    )
    plain = PlainLayout((), tuple((p, vertex) for p in points), (), 0, len(topologies[0]))
    return Case(label, plain, topologies, closed=False)


def two_triangles_and_hexagon() -> Case:
    """Two 2-regular graphs on six points: |Aut_V| = 72 and 12."""
    return _regular(
        "two triangles, hexagon",
        6,
        {
            "two triangles": _cycle([0, 1, 2]) + _cycle([3, 4, 5]),
            "hexagon": _cycle(list(range(6))),
        },
    )


def cube_wagner_two_k4() -> Case:
    """Three 3-regular graphs on eight points: |Aut_V| = 48, 16 and 1152."""
    cube = _cycle([0, 1, 2, 3]) + _cycle([4, 5, 6, 7]) + [(i, i + 4) for i in range(4)]
    wagner = _cycle(list(range(8))) + [(i, i + 4) for i in range(4)]
    return _regular(
        "cube, Wagner, two K4",
        8,
        {
            "cube": cube,
            "Wagner": wagner,
            "two K4": _complete([0, 1, 2, 3]) + _complete([4, 5, 6, 7]),
        },
    )


def twin_c_pairs(pairs: int, m: int) -> Case:
    """``pairs`` pairs of points of one type, each joined by ``m`` parallel
    C: |Aut_V| = pairs! 2^pairs."""
    vertex = _local_vertex("Twin", 0, m)
    edges = [("C", f"a{i}", f"b{i}") for i in range(pairs) for _ in range(m)]
    local = tuple((f"{side}{i}", vertex) for i in range(pairs) for side in "ab")
    plain = PlainLayout((), local, (), *_counts(edges))
    return Case(f"twin C {pairs}x(m={m})", plain, (_props(edges),), closed=False)


def twin_rc_pairs(pairs: int) -> Case:
    """``pairs`` pairs ``(a_i, b_i)`` joined by two parallel ``R(a_i, b_i)``
    and three parallel ``C(a_i, b_i)``: |Aut_V| = pairs!."""
    tail = _local_vertex("TwinTail", 0, 5)
    head = _local_vertex("TwinHead", 2, 3)
    edges = []
    local = []
    for i in range(pairs):
        edges += [("R", f"a{i}", f"b{i}")] * 2 + [("C", f"a{i}", f"b{i}")] * 3
        local += [(f"a{i}", tail), (f"b{i}", head)]
    plain = PlainLayout((), tuple(local), (), *_counts(edges))
    return Case(f"twin R and C {pairs} pairs", plain, (_props(edges),), closed=False)


def tadpole_star(centres: int, tadpoles: int) -> Case:
    """Externals ``x`` and ``y`` joined by C to each centre ``c_i``; each
    centre has ``tadpoles`` points ``t_ij`` with ``R(c_i, t_ij)`` and a C
    loop.  |Aut_V| = centres! (tadpoles!)^centres."""
    centre = _local_vertex("Centre", 0, 2 + tadpoles)
    tad = _local_vertex("Tadpole", 1, 2)
    edges = []
    local = []
    for i in range(centres):
        edges += [("C", "x", f"c{i}"), ("C", "y", f"c{i}")]
        local.append((f"c{i}", centre))
        for j in range(tadpoles):
            edges += [("C", f"t{i}_{j}", f"t{i}_{j}"), ("R", f"c{i}", f"t{i}_{j}")]
            local.append((f"t{i}_{j}", tad))
    plain = PlainLayout(("x", "y"), tuple(local), (), *_counts(edges))
    return Case(f"tadpole star {centres}x{tadpoles}", plain, (_props(edges),), closed=False)


def r_cycle_and_reverse() -> Case:
    """``R(a, b) R(b, c) R(c, a)`` and the reversed cycle, with ``C(x, a)``,
    ``C(y, b)`` and ``C(z, c)``: the externals fix every point, so the two
    are two classes, each with |Aut_V| = 1.  The engine never gives an R
    cycle, so only a key that keeps the R direction tells them apart."""
    vertex = _local_vertex("Cyc", 1, 2)
    c_edges = [("C", "x", "a"), ("C", "y", "b"), ("C", "z", "c")]
    forward = [("R", "a", "b"), ("R", "b", "c"), ("R", "c", "a")] + c_edges
    backward = [("R", b, a) if kind == "R" else (kind, a, b) for kind, a, b in forward]
    plain = PlainLayout(("x", "y", "z"), tuple((p, vertex) for p in "abc"), (), 3, 3)
    return Case("R 3-cycle and its reverse", plain, (_props(forward), _props(backward)),
                closed=False)


def loops_and_two_cycles(m: int) -> Case:
    """``m`` points with ``R(l, l)`` and ``m`` pairs with ``R(a, b) R(b, a)``,
    one local type psi phi: |Aut_V| = m! m! 2^m.  An arc of a node to itself
    and an arc to a node of its own colour look alike to refinement unless
    the self-loop label is part of the signature."""
    vertex = _local_vertex("Loop", 1, 1)
    edges = [("R", f"l{i}", f"l{i}") for i in range(m)]
    for i in range(m):
        edges += [("R", f"a{i}", f"b{i}"), ("R", f"b{i}", f"a{i}")]
    points = [f"l{i}" for i in range(m)] + [f"{s}{i}" for i in range(m) for s in "ab"]
    plain = PlainLayout((), tuple((p, vertex) for p in points), (), *_counts(edges))
    return Case(f"{m} R loops, {m} R 2-cycles", plain, (_props(edges),), closed=False)


def _strongly_regular_16_6_2_2() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """The Shrikhande graph and the 4x4 rook graph on the cells of Z4 x Z4."""
    cells = [(a, b) for a in range(4) for b in range(4)]
    steps = {(0, 1), (0, 3), (1, 0), (3, 0), (1, 1), (3, 3)}
    pairs = [(i, j) for i in range(16) for j in range(i + 1, 16)]
    shrikhande = [
        (i, j)
        for i, j in pairs
        if ((cells[j][0] - cells[i][0]) % 4, (cells[j][1] - cells[i][1]) % 4) in steps
    ]
    rook = [(i, j) for i, j in pairs if cells[i][0] == cells[j][0] or cells[i][1] == cells[j][1]]
    return shrikhande, rook


def strongly_regular_join() -> Case:
    """The complement of the Shrikhande graph beside the 4x4 rook graph, as
    C propagators on 32 points of one type: 400 C, each point of degree 25.
    Both parts are strongly regular with parameters (16, 6, 2, 2), so colour
    refinement splits nothing.  |Aut_V| = 192 * 1152 = 221,184."""
    shrikhande, rook = _strongly_regular_16_6_2_2()
    union = set(shrikhande) | {(i + 16, j + 16) for i, j in rook}
    edges = [("C", f"p{i}", f"p{j}") for i in range(32) for j in range(i + 1, 32)
             if (i, j) not in union]
    vertex = _local_vertex("Deg25", 0, 25)
    plain = PlainLayout((), tuple((f"p{i}", vertex) for i in range(32)), (), 0, len(edges))
    return Case("co(Shrikhande + rook 4x4)", plain, (_props(edges),), closed=False)


def dummy_graph_group_order(pynauty, plain: PlainLayout, props) -> int:
    """Order of the whole colour-preserving automorphism group, from
    pynauty, of the undirected dummy-node graph of ``props`` on a layout of
    local vertices: ``R(a, b)`` as ``a - tail - head - b``, ``C(a, b)`` as
    ``a - c - b`` and ``C(a, a)`` as ``a - c``.  The cells are one per
    external, one per local vertex value, then the tail, head and C dummies.
    Parallel propagators give twin dummies, so the order also counts their
    swaps."""
    assert not plain.nonlocal_instances
    node = {label: i for i, label in enumerate(plain.externals)}
    cells = [{i} for i in range(len(node))]
    types: list[Vertex] = []
    for label, vertex in plain.local_points:
        node[label] = len(node)
        if vertex not in types:
            types.append(vertex)
            cells.append(set())
        cells[len(plain.externals) + types.index(vertex)].add(node[label])
    adjacency: dict[int, list[int]] = {}
    dummy_cells: dict[str, set[int]] = {"tail": set(), "head": set(), "C": set()}
    total = len(node)

    def add_dummy(kind: str) -> int:
        nonlocal total
        dummy_cells[kind].add(total)
        total += 1
        return total - 1

    def link(u: int, w: int) -> None:
        adjacency.setdefault(u, []).append(w)

    for p in props:
        a, b = node[p.spatial_left], node[p.spatial_right]
        if p.kind == "R":
            tail, head = add_dummy("tail"), add_dummy("head")
            link(a, tail)
            link(tail, head)
            link(head, b)
        else:
            c = add_dummy("C")
            link(a, c)
            if b != a:
                link(c, b)
    colouring = cells + [cell for cell in dummy_cells.values() if cell]
    graph = pynauty.Graph(
        total, directed=False, adjacency_dict=adjacency, vertex_coloring=colouring
    )
    _generators, mantissa, exponent, _orbits, _n_orbits = pynauty.autgrp(graph)
    assert exponent == 0 and float(mantissa).is_integer(), (mantissa, exponent)
    return int(mantissa)


@lru_cache(maxsize=None)
def hand_built_cases() -> tuple[Case, ...]:
    return (
        star(6),
        tree(3, 2),
        two_triangles_and_hexagon(),
        cube_wagner_two_k4(),
        twin_c_pairs(3, 5),
        twin_rc_pairs(3),
        r_cycle_and_reverse(),
    )
