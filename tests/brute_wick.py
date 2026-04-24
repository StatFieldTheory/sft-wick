"""Brute-force Wick-contraction reference implementation.

Independent of ``sft_wick`` (no package imports).  Provides the minimum
primitives needed by the deductive tests to cross-check every
transformation in the symbolic pipeline.

Design:
    - An ``Op`` is a lightweight NamedTuple ``(kind, spatial, comp, uid)``
      with ``kind in {'phi', 'psi'}``; it carries exactly the information
      ``contract_pair`` needs, and nothing else.
    - A ``Pairing`` is a ``frozenset`` of ``frozenset({uid_a, uid_b})`` pairs.
      Order within each pair and order across pairs are therefore irrelevant,
      matching the physics definition of a Wick pairing.
    - Propagators are tuples ``('C'|'R', spatial_left, spatial_right,
      comp_left, comp_right)`` with the canonical convention
      ``(phi_spatial, psi_spatial)`` for R propagators (same as
      ``sft_wick.propagators.contract_pair``).

This module is deliberately naive: it enumerates all ``(2n-1)!!`` pairings
and classifies each from scratch.  That is exponential in ``n`` and
therefore only usable up to ``2n <= 10`` or so, but in that regime it is
rigorously correct — which is exactly what a verification oracle needs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
from itertools import combinations, permutations
from math import factorial
from typing import Iterator, NamedTuple


class Op(NamedTuple):
    """A minimal field-operator record.

    ``kind``  : ``'phi'`` (physical) or ``'psi'`` (response).
    ``spatial``: string label of the spatial point (e.g. ``'x'``, ``'y_0'``).
    ``comp``  : component-index label, or ``None`` for scalar fields.
    ``uid``   : unique integer id, used to identify operators in pairings.
    """

    kind: str
    spatial: str
    comp: str | None
    uid: int


# --------------------------------------------------------------------- #
# 1.  Pairing enumeration
# --------------------------------------------------------------------- #


def _pair_set_from_tuple(pairing_tuple: tuple[tuple[int, int], ...]) -> frozenset:
    return frozenset(frozenset(p) for p in pairing_tuple)


def enumerate_pairings(uids: list[int]) -> Iterator[frozenset]:
    """All complete pairings of ``uids`` as ``frozenset`` of ``frozenset`` pairs.

    Yields ``(2n-1)!!`` pairings when ``len(uids) == 2n``.  Empty for odd
    lengths.
    """
    n = len(uids)
    if n == 0:
        yield frozenset()
        return
    if n % 2 != 0:
        return

    first, rest = uids[0], uids[1:]
    for i, partner in enumerate(rest):
        remaining = rest[:i] + rest[i + 1 :]
        for sub in enumerate_pairings(remaining):
            yield sub | {frozenset({first, partner})}


def count_pairings(n_operators: int) -> int:
    """Closed-form ``(2n-1)!!`` for ``2n == n_operators``.  0 for odd inputs."""
    if n_operators == 0:
        return 1
    if n_operators % 2 != 0:
        return 0
    result = 1
    for k in range(1, n_operators, 2):
        result *= k
    return result


# --------------------------------------------------------------------- #
# 2.  Per-pairing classification
# --------------------------------------------------------------------- #


# A propagator is a tuple: (kind, spatial_left, spatial_right, comp_left, comp_right)
Propagator = tuple[str, str, str, str | None, str | None]


def _contract(op1: Op, op2: Op, ito: bool) -> tuple[Propagator, str] | tuple[None, str]:
    """Independent re-implementation of ``contract_pair``.

    Returns ``(propagator, 'ok')`` or ``(None, reason)``.  ``reason`` is
    one of ``'psi-psi'`` or ``'ito'`` when the pair vanishes on contact.
    """
    if op1.kind == "psi" and op2.kind == "psi":
        return None, "psi-psi"

    if op1.kind == "phi" and op2.kind == "phi":
        return (("C", op1.spatial, op2.spatial, op1.comp, op2.comp), "ok")

    # Exactly one phi, one psi -> R(phi_spatial, psi_spatial).
    if op1.kind == "phi":
        phi_op, psi_op = op1, op2
    else:
        phi_op, psi_op = op2, op1

    if ito and phi_op.spatial == psi_op.spatial:
        return None, "ito"

    return (
        ("R", phi_op.spatial, psi_op.spatial, phi_op.comp, psi_op.comp),
        "ok",
    )


def _has_r_cycle(propagators: list[Propagator]) -> bool:
    """DFS cycle detection on directed R edges (ignoring equal-point R's)."""
    adj: dict[str, list[str]] = defaultdict(list)
    for kind, sl, sr, *_ in propagators:
        if kind == "R" and sl != sr:
            adj[sl].append(sr)
    if not adj:
        return False

    WHITE, GRAY, BLACK = 0, 1, 2
    colour: dict[str, int] = {}
    for v, neighbours in adj.items():
        colour.setdefault(v, WHITE)
        for nb in neighbours:
            colour.setdefault(nb, WHITE)

    def dfs(node: str) -> bool:
        colour[node] = GRAY
        for nb in adj.get(node, ()):
            if colour[nb] == GRAY:
                return True
            if colour[nb] == WHITE and dfs(nb):
                return True
        colour[node] = BLACK
        return False

    return any(dfs(v) for v, c in list(colour.items()) if c == WHITE)


def classify_pairing(
    pairing: frozenset,
    op_by_uid: dict[int, Op],
    ito: bool = True,
) -> tuple[bool, str, list[Propagator]]:
    """Classify a pairing and return ``(kept, reason, propagators)``.

    Rules are applied in a strict sequence so ``reason`` pinpoints *why*
    a pairing was rejected:

        1. ``psi-psi`` — any ψ-ψ pair -> vanish.
        2. ``ito``     — any R(x,x) pair under Itô -> vanish.
        3. ``r-loop``  — causal R-loop among surviving R edges -> vanish.
        4. ``ok``      — accepted.

    ``propagators`` is the list on acceptance; the partial list built
    before rejection, on rejection.  Propagator tuples are deterministic
    but unsorted; sort downstream before comparing.
    """
    propagators: list[Propagator] = []
    for pair in pairing:
        u1, u2 = tuple(pair)
        op1, op2 = op_by_uid[u1], op_by_uid[u2]
        prop, reason = _contract(op1, op2, ito=ito)
        if prop is None:
            return False, reason, propagators
        propagators.append(prop)

    if ito and _has_r_cycle(propagators):
        return False, "r-loop", propagators

    return True, "ok", propagators


# --------------------------------------------------------------------- #
# 3.  Topology signature & multiplicity
# --------------------------------------------------------------------- #


def edge_key(prop: Propagator) -> tuple:
    """Canonical multiset-of-edges key for one propagator.

    - ``C`` edges are symmetric; we store the sorted endpoint pair.
    - ``R`` edges are directed; we keep (spatial_left, spatial_right)
      in the canonical (phi, psi) order from ``_contract``.

    Component indices are ignored — this is a *spatial* topology key,
    deliberately coarser than an isomorphism key on the full multi-graph.
    """
    kind, sl, sr, *_ = prop
    if kind == "C":
        a, b = sorted([sl, sr])
        return ("C", a, b)
    return ("R", sl, sr)


def spatial_signature(propagators: list[Propagator]) -> tuple:
    """Sorted multiset of ``edge_key`` — a pairing's spatial topology."""
    return tuple(sorted(edge_key(p) for p in propagators))


def topology_multiplicities(
    ops: list[Op],
    ito: bool = True,
) -> dict[tuple, int]:
    """Count surviving pairings by spatial topology.

    Returns ``{signature: count}`` where ``signature`` is the tuple from
    :func:`spatial_signature`.
    """
    op_by_uid = {op.uid: op for op in ops}
    uids = [op.uid for op in ops]
    counts: Counter = Counter()
    for p in enumerate_pairings(uids):
        kept, _, props = classify_pairing(p, op_by_uid, ito=ito)
        if kept:
            counts[spatial_signature(props)] += 1
    return dict(counts)


def iter_kept_pairings(
    ops: list[Op],
    ito: bool = True,
) -> Iterator[tuple[frozenset, list[Propagator]]]:
    """Yield ``(pairing, propagators)`` for every surviving pairing."""
    op_by_uid = {op.uid: op for op in ops}
    uids = [op.uid for op in ops]
    for p in enumerate_pairings(uids):
        kept, _, props = classify_pairing(p, op_by_uid, ito=ito)
        if kept:
            yield p, props


def classify_all(
    ops: list[Op],
    ito: bool = True,
) -> tuple[set[frozenset], dict[str, int]]:
    """Run classification over every candidate pairing.

    Returns ``(kept_pairings, reason_counts)``.  ``reason_counts`` tallies
    the number of pairings rejected by each rule, summing to the pairing
    count minus ``|kept_pairings|``.  Useful for confirming "rejection A
    fires N times" invariants.
    """
    op_by_uid = {op.uid: op for op in ops}
    uids = [op.uid for op in ops]
    kept: set[frozenset] = set()
    reasons: Counter = Counter()
    for p in enumerate_pairings(uids):
        ok, reason, _ = classify_pairing(p, op_by_uid, ito=ito)
        if ok:
            kept.add(p)
        else:
            reasons[reason] += 1
    reasons["ok"] = len(kept)
    return kept, dict(reasons)


# --------------------------------------------------------------------- #
# 4.  Expected perturbative coefficients
# --------------------------------------------------------------------- #


def multinomial_coefficient(counts: tuple[int, ...]) -> int:
    """``n! / (n_0! n_1! ... n_{k-1}!)`` for ``sum(counts) == n``."""
    n = sum(counts)
    result = factorial(n)
    for c in counts:
        result //= factorial(c)
    return result


def taylor_prefactor(n: int, multinomial: int) -> Fraction:
    """``(-1)^n / n! × multinomial`` as an exact rational.

    This is the coefficient that :class:`DiagramTerm.rational_prefactor`
    should equal, pre-response-phase.  Compare ``.numerator / .denominator``
    for sft-wick's ``Rational`` against the value returned here.
    """
    sign = -1 if n % 2 else 1
    return Fraction(sign * multinomial, factorial(n))


def response_phase(n_R: int) -> complex:
    """``(-i)^{n_R}`` as a complex number."""
    return [1.0, -1j, -1.0, 1j][n_R % 4]


# --------------------------------------------------------------------- #
# 5.  Leg-count vanishing predictions
# --------------------------------------------------------------------- #


def vanishes_by_leg_count(ops: list[Op]) -> bool:
    """``True`` iff no complete ψ-avoiding-ψ pairing exists.

    A pairing is non-vanishing only if every ψ finds a φ partner and the
    remaining φ's pair among themselves.  This requires ``n_psi <= n_phi``
    and ``(n_phi - n_psi)`` even.
    """
    n_phi = sum(1 for op in ops if op.kind == "phi")
    n_psi = sum(1 for op in ops if op.kind == "psi")
    if n_psi > n_phi:
        return True
    if (n_phi - n_psi) % 2 != 0:
        return True
    return False


# --------------------------------------------------------------------- #
# 6.  Convenience: build the Case A / Case B operator lists
# --------------------------------------------------------------------- #


def case_A_operators(uid_start: int = 0) -> list[Op]:
    """Order-2 Case A operators: 2 external φ + 2 × (1ψ + 2φ) cubic vertices.

    External at ``x``, ``y``.  Vertices at ``z0`` and ``z1``.  Scalar (no
    component index).  Order in the returned list is:

        [φ(x), φ(y),
         ψ(z0), φ(z0), φ(z0),
         ψ(z1), φ(z1), φ(z1)]

    with consecutive uids starting at ``uid_start``.  Total 8 operators
    → 105 candidate pairings.
    """
    uid = uid_start
    ops = [
        Op("phi", "x", None, uid + 0),
        Op("phi", "y", None, uid + 1),
        Op("psi", "z0", None, uid + 2),
        Op("phi", "z0", None, uid + 3),
        Op("phi", "z0", None, uid + 4),
        Op("psi", "z1", None, uid + 5),
        Op("phi", "z1", None, uid + 6),
        Op("phi", "z1", None, uid + 7),
    ]
    return ops


def case_B_FK_operators(uid_start: int = 0) -> list[Op]:
    """Order-2 Case B FK operators: 2 external φ + 1 local cubic + 1 non-local ψψψ.

    Order:

        [φ(x), φ(y),
         ψ(z0), φ(z0), φ(z0),      # local F vertex at z0
         ψ(y1), ψ(y2), ψ(y3)]      # non-local K vertex at (y1, y2, y3)

    Total 8 operators.  4 φ and 4 ψ → balanced, pairings survive leg-count.
    """
    uid = uid_start
    ops = [
        Op("phi", "x", None, uid + 0),
        Op("phi", "y", None, uid + 1),
        Op("psi", "z0", None, uid + 2),
        Op("phi", "z0", None, uid + 3),
        Op("phi", "z0", None, uid + 4),
        Op("psi", "y1", None, uid + 5),
        Op("psi", "y2", None, uid + 6),
        Op("psi", "y3", None, uid + 7),
    ]
    return ops


def case_B_KK_operators(uid_start: int = 0) -> list[Op]:
    """Order-2 Case B KK operators: 2 external φ + 2 non-local ψψψ vertices.

    Order:

        [φ(x), φ(y),
         ψ(y1), ψ(y2), ψ(y3),
         ψ(y4), ψ(y5), ψ(y6)]

    Total 8 operators, 2 φ + 6 ψ — must vanish by leg count
    (:func:`vanishes_by_leg_count`) since 2 < 6.
    """
    uid = uid_start
    ops = [
        Op("phi", "x", None, uid + 0),
        Op("phi", "y", None, uid + 1),
    ] + [Op("psi", f"y{k}", None, uid + 2 + k) for k in range(6)]
    return ops
