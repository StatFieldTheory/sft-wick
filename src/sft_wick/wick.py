"""Core Wick contraction engine.

Enumerates all valid pairings (contractions) of field operators and evaluates
each pairing as a product of propagators.
"""

from __future__ import annotations

from collections import Counter
from itertools import chain, groupby, permutations
from operator import itemgetter
from typing import Callable, Iterable, Iterator, Optional, Sequence

from .expressions import Expr, Product, Propagator, Sum, ZERO, canonical_edge
from .fields import FieldOperator, FieldType
from .propagators import contract_pair, propagator

# A pairing is a tuple of (index_i, index_j) pairs
Pairing = tuple[tuple[int, int], ...]

# An R propagator between labels: (phi_label, psi_label)
REdge = tuple[str, str]

_EXHAUSTED = object()


def _label_key(label: str) -> tuple[str, str]:
    """Sort key of a spatial label.

    Strings keep their order.  A label of another type sorts by its type
    first, so an operator list with an int or ``None`` label still contracts.
    """
    return (type(label).__name__, label)


def _walk(depth: int, steps: Callable[[int], Iterator[None]]) -> Iterator[None]:
    """Stop at each complete path of a depth-first search, lazily.

    ``steps(level)`` yields once per choice at *level*, with the choice
    applied, and undoes it when resumed.  The search keeps its own stack, so
    its depth is not bound by the recursion limit.
    """
    if depth == 0:
        yield
        return
    stack = [steps(0)]
    while stack:
        if next(stack[-1], _EXHAUSTED) is _EXHAUSTED:
            stack.pop()
        elif len(stack) == depth:
            yield
        else:
            stack.append(steps(len(stack)))


def r_vanishes(edges: Iterable[REdge]) -> bool:
    r"""Whether a product of R propagators vanishes.

    The one vanishing rule of the Wick engines, for operators and for
    labels alike: the edges close a directed cycle.  A cycle
    :math:`R(a_1,a_2)R(a_2,a_3)\dots R(a_n,a_1)` would need
    :math:`t_1>t_2>\dots>t_n>t_1`, which no causal time ordering satisfies.
    Its shortest form is one equal-point edge, the Itô rule
    :math:`\Theta(0)=0` that gives :math:`R(x,x)=0`.  The rule depends on the
    labels only, and adding edges never undoes it.

    Args:
        edges: ``(phi_label, psi_label)`` pairs.
    """
    adj: dict[str, list[str]] = {}
    incoming: dict[str, int] = {}
    for sl, sr in edges:
        adj.setdefault(sl, []).append(sr)
        incoming[sr] = incoming.get(sr, 0) + 1
    if adj.keys().isdisjoint(incoming):
        return False
    # Remove the points no remaining edge enters; a cycle never empties.
    free = [node for node in adj if node not in incoming]
    while free:
        for nb in adj.get(free.pop(), ()):
            incoming[nb] -= 1
            if not incoming[nb]:
                del incoming[nb]
                free.append(nb)
    return bool(incoming)


def generate_all_pairings(indices: list[int]) -> Iterator[Pairing]:
    """Generate all complete pairings of the given indices.

    For 2n items, yields (2n-1)!! = 1*3*5*...*(2n-1) pairings.
    Each pairing is a tuple of n pairs.
    """
    n = len(indices)
    if n == 0:
        yield ()
        return
    if n % 2 != 0:
        return  # odd count -> no complete pairing

    first = indices[0]
    rest = indices[1:]
    for i, partner in enumerate(rest):
        remaining = rest[:i] + rest[i + 1 :]
        for sub_pairing in generate_all_pairings(remaining):
            yield ((first, partner),) + sub_pairing


def r_classes(
    phi_labels: Sequence[str],
    psi_labels: Sequence[str],
) -> Iterator[tuple[REdge, ...]]:
    """Enumerate the R assignments of a contraction, at the label level.

    This is the one enumeration of R assignments at the label level:
    :func:`_lift_r` turns a class into its operator assignments, and
    ``perturbation._enumerate_component_routings`` does so for a topology,
    with the operators at observable points held fixed.  Each ψ takes a φ with remaining
    capacity at its label.  A class is one multiset of
    ``(phi_label, psi_label)`` edges, so the operators at a label do not
    multiply it: consecutive ψ's at one label take non-decreasing φ labels.
    A branch stops at the first edge :func:`r_vanishes` rejects.  Yields
    nothing when ψ's outnumber φ's or the φ's left over cannot pair.
    """
    if len(psi_labels) > len(phi_labels):
        return
    if (len(phi_labels) - len(psi_labels)) % 2 != 0:
        return

    capacity = Counter(phi_labels)
    phi_order = sorted(capacity, key=_label_key)
    phi_rank = {label: rank for rank, label in enumerate(phi_order)}
    psi_list = sorted(psi_labels, key=_label_key)
    edges: list[REdge] = []

    def place(idx: int) -> Iterator[None]:
        psi_label = psi_list[idx]
        repeated = idx > 0 and psi_label == psi_list[idx - 1]
        lowest = phi_rank[edges[-1][0]] if repeated else 0
        for phi_label in phi_order[lowest:]:
            if capacity[phi_label] <= 0:
                continue
            edges.append((phi_label, psi_label))
            if not r_vanishes(edges):
                capacity[phi_label] -= 1
                yield
                capacity[phi_label] += 1
            edges.pop()

    for _ in _walk(len(psi_list), place):
        yield tuple(edges)


def _distinct_orders(items: tuple[str, ...]) -> Iterator[tuple[str, ...]]:
    """The distinct orderings of a multiset, lazily."""
    remaining = Counter(items)
    distinct = sorted(remaining, key=_label_key)
    order: list[str] = []

    def place(_level: int) -> Iterator[None]:
        for item in distinct:
            if remaining[item]:
                remaining[item] -= 1
                order.append(item)
                yield
                order.pop()
                remaining[item] += 1

    for _ in _walk(len(items), place):
        yield tuple(order)


def _orderings(runs: Sequence[tuple[str, ...]]) -> Iterator[tuple[str, ...]]:
    """Each run in each of its distinct orders, concatenated, lazily.

    ``itertools.product`` would hold every order of every run in memory.
    """
    chosen: list[tuple[str, ...]] = [()] * len(runs)

    def place(level: int) -> Iterator[None]:
        for order in _distinct_orders(runs[level]):
            chosen[level] = order
            yield

    for _ in _walk(len(runs), place):
        yield tuple(chain.from_iterable(chosen))


def _fill(
    chosen: list[int],
    pools: Sequence[tuple[list[int], list[int]]],
) -> Iterator[tuple[int, ...]]:
    """*chosen* with each ``(positions, operators)`` of *pools* filled in
    each way, lazily."""
    def place(level: int) -> Iterator[None]:
        positions, operators = pools[level]
        for pick in permutations(operators, len(positions)):
            for position, op in zip(positions, pick):
                chosen[position] = op
            yield

    for _ in _walk(len(pools), place):
        yield tuple(chosen)


def _lift_r(
    r_assign: Sequence[REdge],
    phi_at: dict[str, list[int]],
) -> Iterator[tuple[int, ...]]:
    """The operator assignments of one R class of :func:`r_classes`, lazily.

    Each is the φ operator of every edge, position by position.  A ψ label
    may order its φ labels in any distinct way, and the operators at a φ
    label fill its positions in any order.  Every combination is a different
    assignment of the class, and none is a dead end.
    """
    runs = [
        tuple(phi for phi, _ in run)
        for _, run in groupby(r_assign, key=itemgetter(1))
    ]
    for labels in _orderings(runs):
        positions: dict[str, list[int]] = {}
        for position, label in enumerate(labels):
            positions.setdefault(label, []).append(position)
        pools = [(at, phi_at[label]) for label, at in positions.items()]
        yield from _fill([0] * len(labels), pools)


def _indices_at(
    indices: Sequence[int], operators: Sequence[FieldOperator]
) -> dict[str, list[int]]:
    """``indices`` grouped by the spatial label of their operators."""
    at: dict[str, list[int]] = {}
    for idx in indices:
        at.setdefault(operators[idx].spatial_arg, []).append(idx)
    return at


def generate_valid_pairings(
    phi_indices: list[int],
    psi_indices: list[int],
    operators: Sequence[FieldOperator],
) -> Iterator[Pairing]:
    """Generate only non-vanishing pairings (no psi-psi contractions).

    Exploits the MSR structure: each psi must pair with a phi (producing R),
    and remaining phi's pair among themselves (producing C).  An R part
    comes from :func:`r_classes`, which stops at the first edge
    :func:`r_vanishes` rejects, and :func:`_lift_r`; every step is lazy.  A
    pairing lists its R pairs in the order of *psi_indices*.  The pairings
    come class by class; :func:`contracted_pairings` puts them in operator
    order.
    """
    phi_at = _indices_at(phi_indices, operators)
    psi_at = _indices_at(psi_indices, operators)

    # The ψ of each position of a class, and the positions in the order of
    # psi_indices.
    psi_by_position = [
        idx for s in sorted(psi_at, key=_label_key) for idx in psi_at[s]
    ]
    position_of = {idx: k for k, idx in enumerate(psi_by_position)}
    output_order = [position_of[idx] for idx in psi_indices]
    phi_labels = [operators[idx].spatial_arg for idx in phi_indices]
    psi_labels = [operators[idx].spatial_arg for idx in psi_indices]
    # With one operator at each label a class is one assignment.
    one_each = len(phi_at) == len(phi_labels) and len(psi_at) == len(psi_labels)

    for r_assign in r_classes(phi_labels, psi_labels):
        lifts = (
            ([phi_at[phi][0] for phi, _ in r_assign],)
            if one_each
            else _lift_r(r_assign, phi_at)
        )
        for chosen in lifts:
            r_pairs = tuple(
                (chosen[k], psi_by_position[k]) for k in output_order
            )
            taken = set(chosen)
            remaining = [idx for idx in phi_indices if idx not in taken]
            for c_pairing in generate_all_pairings(remaining):
                yield r_pairs + c_pairing


def evaluate_pairing(
    operators: Sequence[FieldOperator],
    pairing: Pairing,
) -> Optional[tuple[Expr, list[Propagator]]]:
    r"""Evaluate a single complete pairing.

    Returns ``(Product of propagators, list of individual propagators)``
    or ``None`` if any contraction vanishes.  The engines do not call it,
    since their enumeration stops at a vanishing edge; it checks one pairing
    on its own, as the tests do against the full list of pairings.

    Vanishing conditions:

    - Contraction of two :math:`\psi` operators
    - The R propagators meet :func:`r_vanishes`: equal-point
      :math:`R(x,x)=0`, or a directed cycle among their spatial arguments
      (e.g. :math:`R(a,b)\,R(b,a)=0`)

    Args:
        operators: The full list of field operators.
        pairing: Tuple of ``(i, j)`` index pairs.
    """
    propagators: list[Propagator] = []
    for i, j in pairing:
        prop = contract_pair(operators[i], operators[j])
        if prop is None:
            return None
        propagators.append(prop)

    if len(propagators) == 0:
        return None

    r_edges = (
        (p.spatial_left, p.spatial_right) for p in propagators if p.kind == "R"
    )
    if r_vanishes(r_edges):
        return None

    return _product_of(propagators), propagators


def _product_of(propagators: list[Propagator]) -> Expr:
    """The product of the propagators of one pairing."""
    if len(propagators) == 1:
        return propagators[0]
    return Product(tuple(propagators))


def contracted_pairings(
    operators: Sequence[FieldOperator],
    pairings: Iterable[Pairing],
) -> Iterator[tuple[Pairing, Expr, list[Propagator]]]:
    """Each pairing of :func:`generate_valid_pairings` with its propagators.

    The pairings come in the order of their φ operators, not of the R
    classes, so the expression and the first pairing of a diagram do not
    depend on how the labels sort.  The enumeration has applied
    :func:`r_vanishes`, so nothing here checks it again.  An operator pair
    recurs across pairings and is contracted once: the frozen propagator is
    shared.  The empty pairing has no product and is skipped.
    """
    n_r = sum(op.is_response for op in operators)

    def operator_order(pairing: Pairing) -> tuple[list[int], list[int]]:
        phis = [phi for phi, _ in pairing[:n_r]]
        return sorted(phis), phis

    contracted: dict[tuple[int, int], Propagator] = {}
    for pairing in sorted(pairings, key=operator_order):
        if not pairing:
            continue
        propagators = []
        for pair in pairing:
            prop = contracted.get(pair)
            if prop is None:
                prop = contract_pair(operators[pair[0]], operators[pair[1]])
                contracted[pair] = prop
            propagators.append(prop)
        yield pairing, _product_of(propagators), propagators


def wick_contract(
    operators: Sequence[FieldOperator],
    *,
    keep: Optional[Callable[[Pairing], bool]] = None,
) -> tuple[Expr, list[Pairing]]:
    r"""Apply Wick's theorem to a product of field operators.

    Response propagators at equal spatial points vanish, the Itô rule
    :math:`R(x,x)=0`, and so does a causal loop of R.

    Args:
        operators: Sequence of field operators to contract.
        keep: When given, only the pairings it accepts are contracted.

    Returns:
        ``(expression, surviving_pairings)`` where *expression* is a
        Sum of Products of Propagators, and *surviving_pairings* lists
        the non-zero pairings.
    """
    phi_indices = [i for i, op in enumerate(operators) if op.is_physical]
    psi_indices = [i for i, op in enumerate(operators) if op.is_response]
    pairings = generate_valid_pairings(phi_indices, psi_indices, operators)

    terms: list[Expr] = []
    surviving_pairings: list[Pairing] = []

    for pairing, expr, _props in contracted_pairings(
        operators, pairings if keep is None else filter(keep, pairings)
    ):
        terms.append(expr)
        surviving_pairings.append(pairing)

    if not terms:
        return ZERO, []
    if len(terms) == 1:
        return terms[0], surviving_pairings

    return Sum(tuple(terms)), surviving_pairings


# Spatial signature: sorted tuple of canonical (kind, spatial_left, spatial_right)
SpatialSignature = tuple[tuple[str, str, str], ...]


# ---------------------------------------------------------------------------
# Spatial-level Wick contraction engine
# ---------------------------------------------------------------------------


def _enumerate_c_pairings(
    remaining: dict[str, int],
    _min_cross_partner: str | None = None,
) -> Iterator[list[tuple[str, str]]]:
    """Enumerate ways to pair remaining φ-slots into C propagators.

    Always picks from the lexicographically first available point.  When
    a point has multiple remaining slots going to cross-edges,
    ``_min_cross_partner`` enforces non-decreasing partner ordering.

    One spatial pairing can still come twice, with a self-loop before or
    after a cross-edge: three slots at x and one at y give
    ``[(x, x), (x, y)]`` and ``[(x, y), (x, x)]``.  Callers key by
    signature.
    """
    # Find first point with remaining > 0
    first_pt: str | None = None
    for pt in sorted(remaining.keys()):
        if remaining[pt] > 0:
            first_pt = pt
            break

    if first_pt is None:
        yield []
        return

    remaining[first_pt] -= 1

    # Self-loop (no ordering constraint; self-loops are distinct from cross)
    if remaining[first_pt] > 0:
        remaining[first_pt] -= 1
        for sub in _enumerate_c_pairings(remaining):
            yield [(first_pt, first_pt)] + sub
        remaining[first_pt] += 1

    # Cross-edge to another point
    for other_pt in sorted(remaining.keys()):
        if other_pt == first_pt:
            continue
        if remaining[other_pt] <= 0:
            continue
        if _min_cross_partner is not None and other_pt < _min_cross_partner:
            continue

        remaining[other_pt] -= 1

        # If first_pt still has remaining, enforce non-decreasing partner
        next_min = other_pt if remaining[first_pt] > 0 else None

        for sub in _enumerate_c_pairings(remaining, next_min):
            yield [(first_pt, other_pt)] + sub
        remaining[other_pt] += 1

    remaining[first_pt] += 1


def _operators_at(
    operators: Sequence[FieldOperator],
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """φ and ψ operator indices at each spatial label."""
    phi = [i for i, op in enumerate(operators) if op.field_type == FieldType.PHYSICAL]
    psi = [i for i, op in enumerate(operators) if op.field_type != FieldType.PHYSICAL]
    return _indices_at(phi, operators), _indices_at(psi, operators)


def _r_classes_with_leftovers(
    phi_at: dict[str, list[int]],
    psi_at: dict[str, list[int]],
) -> Iterator[tuple[tuple[REdge, ...], dict[str, int]]]:
    """Each R class with the φ-capacity it leaves for C propagators."""
    phi_labels = [s for s, ops in phi_at.items() for _ in ops]
    psi_labels = [s for s, ops in psi_at.items() for _ in ops]
    for r_assign in r_classes(phi_labels, psi_labels):
        remaining = {s: len(ops) for s, ops in phi_at.items()}
        for phi_s, _psi_s in r_assign:
            remaining[phi_s] -= 1
        yield r_assign, remaining


def _representative(
    operators: Sequence[FieldOperator],
    phi_at: dict[str, list[int]],
    psi_at: dict[str, list[int]],
    r_assign: Sequence[REdge],
    c_pairing: Sequence[tuple[str, str]],
) -> tuple[list[Propagator], Pairing]:
    """One operator-level pairing of a spatial topology, with its
    propagators: the operators at each label are taken in order."""
    # Track which operators have been consumed at each point
    phi_used: dict[str, int] = {s: 0 for s in phi_at}
    psi_used: dict[str, int] = {s: 0 for s in psi_at}

    props: list[Propagator] = []
    pairs: list[tuple[int, int]] = []

    for phi_s, psi_s in r_assign:
        phi_idx = phi_at[phi_s][phi_used[phi_s]]
        psi_idx = psi_at[psi_s][psi_used[psi_s]]
        phi_used[phi_s] += 1
        psi_used[psi_s] += 1
        props.append(propagator("R", operators[phi_idx], operators[psi_idx]))
        pairs.append((phi_idx, psi_idx))

    for s1, s2 in c_pairing:
        idx1 = phi_at[s1][phi_used[s1]]
        phi_used[s1] += 1
        idx2 = phi_at[s2][phi_used[s2]]
        phi_used[s2] += 1
        props.append(propagator("C", operators[idx1], operators[idx2]))
        pairs.append((idx1, idx2))

    return props, tuple(pairs)


def wick_contract_spatial(
    operators: Sequence[FieldOperator],
) -> dict[SpatialSignature, tuple[list[Propagator], Pairing]]:
    r"""Spatial-level Wick contraction.

    Enumerates spatial topologies (not individual operator-level pairings)
    with one representative pairing each.  This avoids the combinatorial
    explosion from component-index routing; the routings of a topology
    (:func:`sft_wick.perturbation._enumerate_component_routings`) are its
    operator-level pairings.

    Args:
        operators: Sequence of field operators.

    Returns:
        Dict mapping each :data:`SpatialSignature` to
        ``(reference_propagators, representative_pairing)``.
    """
    if not operators:
        return {}

    phi_at, psi_at = _operators_at(operators)
    results: dict[SpatialSignature, tuple[list[Propagator], Pairing]] = {}

    for r_assign, remaining in _r_classes_with_leftovers(phi_at, psi_at):
        for c_pairing in _enumerate_c_pairings(remaining):
            # Build canonical spatial signature
            sig_edges: list[tuple[str, str, str]] = []
            for phi_s, psi_s in r_assign:
                sig_edges.append(("R", phi_s, psi_s))
            for s1, s2 in c_pairing:
                sig_edges.append(canonical_edge("C", s1, s2))
            sig: SpatialSignature = tuple(sorted(sig_edges))

            # _enumerate_c_pairings can yield one C pairing twice.
            if sig not in results:
                results[sig] = _representative(
                    operators, phi_at, psi_at, r_assign, c_pairing
                )

    return results


def r_class_pairings(
    operators: Sequence[FieldOperator],
) -> Iterator[tuple[list[Propagator], Pairing]]:
    """One complete pairing per R class of ``operators``.

    What a contraction joins through R is set by the class, and every class
    has a C pairing of the φ's it leaves, so this holds every R of
    :func:`wick_contract` without its C pairings.
    """
    if not operators:
        return
    phi_at, psi_at = _operators_at(operators)
    for r_assign, remaining in _r_classes_with_leftovers(phi_at, psi_at):
        c_pairing = next(_enumerate_c_pairings(remaining))
        yield _representative(operators, phi_at, psi_at, r_assign, c_pairing)
