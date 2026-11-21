"""Counts of valid operator pairings, from plain data.

This module imports the standard library only and nothing from the
package, so its counts can be compared with those of the contraction
engine.

A pairing joins every operator to one partner.  It is valid when

* no psi is joined to a psi,
* no psi is joined to a phi on its own point,
* the response propagators form no directed cycle on the points, where a
  psi on point ``a`` joined to a phi on point ``b`` is the edge ``a -> b``.

Two phi may always be joined, on one point or on two.

The input describes groups of operators.  Entry ``k`` of ``points`` is the
label of the point of group ``k`` and ``phi_counts[k]`` its number of phi.
``psi_owner`` has one entry per psi, the index of its group.  A label may
appear in several groups: the rules above read labels, so groups on one
label are one point.  ``owner_of_group[k]`` names the node that owns group
``k`` for the connected target.  Labels and owners compare by equality, so
``1``, ``1.0`` and ``True`` are one label.  The contraction engine makes each
external operator one node and each vertex instance one node, so a vertex
whose legs sit on separate points is several groups with one owner, and an
external operator is a group of its own.

Two algorithms count the pairings of the ``'moment'`` target.

Direct enumeration assigns to each psi the point of its partner, visits
every assignment without a cycle once, and weights it by the number of
ways to choose distinct phi on the chosen points, a product of falling
factorials.  The ``K`` phi left over pair among themselves in ``(K-1)!!``
ways.

The subset recursion removes a set of points that receive no psi and sends
their psi to the remaining points, with alternating signs over the removed
set (inclusion and exclusion on the sources of an acyclic graph).  Its
state is the set of remaining points with the number of phi already taken
on each, the in-degree vector.

The ``'connected'`` target counts the valid pairings that join all owners
into one component.  With ``total(S)`` the moment count of the operators
owned by ``S``,

    total(S) = sum over subsets T of S that contain the lowest owner of S
               of connected(T) * total(S - T).

Neither algorithm bounds its cost.  Direct enumeration grows like the
number of points to the power of the number of psi, the subset recursion
exponentially in the number of points, and the connected target visits
about ``3**n`` subsets of ``n`` owners.  A caller limits the size of what
it passes.
"""
from __future__ import annotations

from collections.abc import Hashable, Iterator, Sequence
from math import comb

TARGETS = ("moment", "connected")
METHODS = ("auto", "direct", "dp")

#: ``method='auto'`` enumerates directly up to this many psi and takes the
#: subset recursion above it.  The value was set without a fit to timings.
#: Measured on the development machine: on five points the recursion takes
#: 0.02 s at 12 psi and 0.04 s at 16, against 0.4 s and 7 s for direct
#: enumeration; on ten points with 8 psi (cubic order 8) direct enumeration
#: takes 7 s and the recursion 25 s.
DIRECT_MAX_PSI = 12


class UnsupportedLayout(ValueError):
    """A layout whose connected count the owner-subset inclusion-exclusion
    cannot take, under every ``method``: two or more labels with both phi
    and psi are each shared by several owners, so a response cycle could
    cross two components."""


#: Operators of one owner on one point: (point, owner, n_phi, n_psi), with
#: points and owners as indices.
_Group = tuple[int, int, int, int]


def count_valid_pairings(
    points: Sequence[Hashable],
    phi_counts: Sequence[int],
    psi_owner: Sequence[int],
    *,
    owner_of_group: Sequence[Hashable] | None = None,
    target: str = "moment",
    method: str = "auto",
) -> int:
    """Number of valid pairings of the operators described by the input.

    ``points``: the point label of each group of operators.
    ``phi_counts``: the number of phi in each group.
    ``psi_owner``: for each psi, the index of its group.
    ``owner_of_group``: the owner node of each group; each group is its own
    owner when omitted.  Read by ``target='connected'`` only.

    The sequences are lists or tuples, and counts and indices are ``int``
    (``bool`` is refused).  A numpy array or a numpy integer raises
    ``ValueError``; convert with ``tolist()`` first.
    ``target``: ``'moment'`` counts all valid pairings, ``'connected'``
    those that join all owners.  With no owner the connected count is 0.
    ``method``: ``'direct'``, ``'dp'`` (the subset recursion) or ``'auto'``.

    Raises ``ValueError`` on malformed input, and
    :class:`UnsupportedLayout`, a ``ValueError``, for
    ``target='connected'`` when two or more labels that carry both phi and
    psi are each shared by several owners.  The connected count is 0
    without that check when the operators are odd in number or form fewer
    pairs than the owners less one, since a connected graph on ``k`` nodes
    has at least ``k - 1`` edges.
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    groups = _groups(points, phi_counts, psi_owner, owner_of_group)
    if target == "moment":
        return _count_moment(groups, method)
    return _count_connected(groups, method)


# --------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------- #


def _sequence(value: object, name: str) -> Sequence:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            f"{name} must be a sequence (list or tuple), got "
            f"{type(value).__name__}"
        )
    return value


def _index_of(values: Sequence[Hashable], name: str) -> list[int]:
    """Each value replaced by the rank of its first appearance."""
    ranks: dict[Hashable, int] = {}
    out = []
    for position, value in enumerate(values):
        try:
            out.append(ranks.setdefault(value, len(ranks)))
        except TypeError:
            raise ValueError(
                f"{name}[{position}] must be hashable, got "
                f"{type(value).__name__}"
            ) from None
    return out


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _psi_per_group(psi_owner: Sequence[int], n_groups: int) -> list[int]:
    counts = [0] * n_groups
    for position, group in enumerate(_sequence(psi_owner, "psi_owner")):
        if not _is_count(group) or not 0 <= group < n_groups:
            raise ValueError(
                f"psi_owner[{position}] must be an index into points "
                f"(0 to {n_groups - 1}), got {group!r}"
            )
        counts[group] += 1
    return counts


def _groups(
    points: Sequence[Hashable],
    phi_counts: Sequence[int],
    psi_owner: Sequence[int],
    owner_of_group: Sequence[Hashable] | None,
) -> tuple[_Group, ...]:
    """Validate the input and return it as index tuples."""
    labels = _index_of(_sequence(points, "points"), "points")
    n_groups = len(labels)
    phi = _sequence(phi_counts, "phi_counts")
    if len(phi) != n_groups:
        raise ValueError(
            f"phi_counts has {len(phi)} entries for {n_groups} points"
        )
    for position, value in enumerate(phi):
        if not _is_count(value) or value < 0:
            raise ValueError(
                f"phi_counts[{position}] must be a non-negative integer, "
                f"got {value!r}"
            )
    psi = _psi_per_group(psi_owner, n_groups)
    if owner_of_group is None:
        owners = list(range(n_groups))
    else:
        named = _sequence(owner_of_group, "owner_of_group")
        if len(named) != n_groups:
            raise ValueError(
                f"owner_of_group has {len(named)} entries for {n_groups} "
                "points"
            )
        owners = _index_of(named, "owner_of_group")
    return tuple(zip(labels, owners, phi, psi))


# --------------------------------------------------------------------- #
# Moment target
# --------------------------------------------------------------------- #


def _perfect_matchings(n_items: int) -> int:
    """``(n-1)!!`` for even ``n >= 0``; 0 for odd or negative ``n``."""
    if n_items < 0 or n_items % 2:
        return 0
    result = 1
    for odd in range(1, n_items, 2):
        result *= odd
    return result


def _falling(n: int, k: int) -> int:
    """``n (n-1) ... (n-k+1)``."""
    result = 1
    for factor in range(n, n - k, -1):
        result *= factor
    return result


def _by_point(groups: Sequence[_Group]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """phi and psi counts per label, for the labels that carry operators."""
    phi: dict[int, int] = {}
    psi: dict[int, int] = {}
    for label, _, n_phi, n_psi in groups:
        phi[label] = phi.get(label, 0) + n_phi
        psi[label] = psi.get(label, 0) + n_psi
    kept = [label for label in phi if phi[label] or psi[label]]
    return tuple(phi[k] for k in kept), tuple(psi[k] for k in kept)


def _count_moment(groups: Sequence[_Group], method: str) -> int:
    phi, psi = _by_point(groups)
    matchings = _perfect_matchings(sum(phi) - sum(psi))
    if matchings == 0:
        return 0
    if method == "auto":
        method = "direct" if sum(psi) <= DIRECT_MAX_PSI else "dp"
    if method == "direct":
        return matchings * _count_direct(phi, psi)
    return matchings * _count_dp(phi, psi)


def _reaches(successors: tuple[int, ...], start: int, goal: int) -> bool:
    """Whether a directed path leads from ``start`` to ``goal``.

    ``successors[a]`` is the bit set of the points ``b`` with an edge
    ``a -> b``.
    """
    seen = 1 << start
    stack = [start]
    while stack:
        mask = successors[stack.pop()] & ~seen
        seen |= mask
        while mask:
            low = mask & -mask
            stack.append(low.bit_length() - 1)
            mask ^= low
    return bool(seen >> goal & 1)


def _replaced(values: tuple[int, ...], position: int, value: int) -> tuple[int, ...]:
    return values[:position] + (value,) + values[position + 1:]


def _count_direct(phi: tuple[int, ...], psi: tuple[int, ...]) -> int:
    """Assignments of every psi to a phi on another point, without a
    directed cycle on the points."""
    sources = [point for point, n in enumerate(psi) for _ in range(n)]
    n_points = len(phi)

    def assign(k: int, free: tuple[int, ...], successors: tuple[int, ...]) -> int:
        if k == len(sources):
            return 1
        own = sources[k]
        total = 0
        for target in range(n_points):
            if target == own or free[target] == 0:
                continue
            if _reaches(successors, target, own):
                continue
            total += free[target] * assign(
                k + 1,
                _replaced(free, target, free[target] - 1),
                _replaced(successors, own, successors[own] | 1 << target),
            )
        return total

    return assign(0, phi, (0,) * n_points)


def _bits(mask: int) -> list[int]:
    return [k for k in range(mask.bit_length()) if mask >> k & 1]


def _submasks(mask: int) -> Iterator[int]:
    """The non-empty subsets of ``mask``."""
    sub = mask
    while sub:
        yield sub
        sub = (sub - 1) & mask


def _placements(
    n_psi: int, targets: Sequence[int], free: tuple[int, ...]
) -> Iterator[tuple[tuple[int, ...], int]]:
    """Ways to join ``n_psi`` distinct psi to distinct phi on ``targets``.

    Yields ``(free phi left on each target, number of ways)`` for each
    in-degree vector on ``targets``.
    """
    if not targets:
        if n_psi == 0:
            yield (), 1
        return
    first, rest = targets[0], targets[1:]
    for taken in range(min(n_psi, free[first]) + 1):
        ways = comb(n_psi, taken) * _falling(free[first], taken)
        for left, more in _placements(n_psi - taken, rest, free):
            yield (free[first] - taken,) + left, ways * more


def _spread(
    n_points: int, targets: Sequence[int], left: tuple[int, ...]
) -> tuple[int, ...]:
    """``left[k]`` at point ``targets[k]`` and 0 at every other point."""
    spread = [0] * n_points
    for point, value in zip(targets, left):
        spread[point] = value
    return tuple(spread)


def _count_dp(phi: tuple[int, ...], psi: tuple[int, ...]) -> int:
    """The same count as :func:`_count_direct`, by removing sets of points
    that receive no psi."""
    n_points = len(phi)
    memo: dict[tuple[int, tuple[int, ...]], int] = {}

    def acyclic(mask: int, free: tuple[int, ...]) -> int:
        """``free[k]`` is the phi left on point ``k``; 0 off ``mask``."""
        if mask == 0:
            return 1
        key = (mask, free)
        if key in memo:
            return memo[key]
        total = 0
        for removed in _submasks(mask):
            rest = mask ^ removed
            targets = _bits(rest)
            n_out = sum(psi[k] for k in _bits(removed))
            sign = 1 if removed.bit_count() % 2 else -1
            for left, ways in _placements(n_out, targets, free):
                spread = _spread(n_points, targets, left)
                total += sign * ways * acyclic(rest, spread)
        memo[key] = total
        return total

    return acyclic((1 << n_points) - 1, phi)


# --------------------------------------------------------------------- #
# Connected target
# --------------------------------------------------------------------- #


def _require_cycles_within_components(groups: Sequence[_Group]) -> None:
    """Refuse layouts where a response cycle can cross two components.

    A cycle passes only through labels with both phi and psi, and it can
    leave the owners of one component only at a label shared with another
    owner, which it must do at two labels or more.
    """
    owners: dict[int, set[int]] = {}
    phi: dict[int, int] = {}
    psi: dict[int, int] = {}
    for label, owner, n_phi, n_psi in groups:
        if n_phi or n_psi:
            owners.setdefault(label, set()).add(owner)
        phi[label] = phi.get(label, 0) + n_phi
        psi[label] = psi.get(label, 0) + n_psi
    crossing = [label for label, named in owners.items()
                if len(named) > 1 and phi[label] and psi[label]]
    if len(crossing) > 1:
        raise UnsupportedLayout(
            "target='connected' is not supported when two or more labels "
            "that carry both phi and psi are each shared by several "
            f"owners ({len(crossing)} such labels): a response cycle "
            "could cross two components"
        )


def _count_connected(groups: Sequence[_Group], method: str) -> int:
    n_owners = len({owner for _, owner, _, _ in groups})
    n_operators = sum(n_phi + n_psi for _, _, n_phi, n_psi in groups)
    # A pairing has one edge per pair, and a connected graph on n_owners
    # nodes has at least n_owners - 1 edges.
    if n_operators % 2 or n_operators // 2 < n_owners - 1:
        return 0
    _require_cycles_within_components(groups)
    if n_owners == 0:
        return 0
    totals: dict[int, int] = {}
    connected: dict[int, int] = {}

    def total(mask: int) -> int:
        if mask not in totals:
            inside = [g for g in groups if mask >> g[1] & 1]
            totals[mask] = _count_moment(inside, method)
        return totals[mask]

    def joined(mask: int) -> int:
        """Valid pairings of the owners in ``mask`` with one component."""
        if mask in connected:
            return connected[mask]
        lowest = mask & -mask
        others = mask ^ lowest
        value = total(mask)
        for split_off in _submasks(others):
            value -= joined(mask ^ split_off) * total(split_off)
        connected[mask] = value
        return value

    return joined((1 << n_owners) - 1)
