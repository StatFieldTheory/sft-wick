"""Core Wick contraction engine.

Enumerates all valid pairings (contractions) of field operators and evaluates
each pairing as a product of propagators.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from math import factorial
from typing import Iterator, Optional, Sequence

from .expressions import Expr, Product, Propagator, Sum, ZERO
from .fields import FieldOperator, FieldType
from .propagators import contract_pair

# A pairing is a tuple of (index_i, index_j) pairs
Pairing = tuple[tuple[int, int], ...]

# Cache for R-cycle detection keyed by spatial R-edge structure
_r_cycle_cache: dict[frozenset[tuple[str, str]], bool] = {}


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


def generate_valid_pairings(
    phi_indices: list[int],
    psi_indices: list[int],
    operators: Sequence[FieldOperator] | None = None,
    ito: bool = False,
) -> Iterator[Pairing]:
    """Generate only non-vanishing pairings (no psi-psi contractions).

    Exploits the MSR structure: each psi must pair with a phi (producing R),
    and remaining phi's pair among themselves (producing C).

    This avoids generating the many zero-valued psi-psi pairings.

    When *operators* and *ito* are provided, also skips phi-psi pairs
    at the same spatial point (R(x,x)=0 under Itô) and phi-psi
    assignments whose R edges form a causal cycle.
    """
    n_phi = len(phi_indices)
    n_psi = len(psi_indices)

    # Quick checks
    if n_psi > n_phi:
        return  # not enough phi's
    if (n_phi - n_psi) % 2 != 0:
        return  # remaining phi's can't pair

    if n_psi == 0:
        # All physical fields: pair them all
        yield from generate_all_pairings(phi_indices)
        return

    # Precompute spatial args for Itô pruning
    do_ito_prune = ito and operators is not None
    if do_ito_prune:
        psi_spatial = [operators[psi_indices[j]].spatial_arg for j in range(n_psi)]

    # Choose which phi's pair with the psi's
    for phi_subset_indices in combinations(range(n_phi), n_psi):
        chosen_phis = [phi_indices[k] for k in phi_subset_indices]
        remaining_phis = [
            phi_indices[k] for k in range(n_phi) if k not in phi_subset_indices
        ]

        # All ways to match psi's to chosen phi's
        for perm in permutations(range(n_psi)):
            # Early Itô: skip if any phi-psi pair shares a spatial point
            if do_ito_prune:
                skip = False
                for j in range(n_psi):
                    phi_spatial = operators[chosen_phis[perm[j]]].spatial_arg
                    if phi_spatial == psi_spatial[j]:
                        skip = True
                        break
                if skip:
                    continue

                # Early R-cycle check on the phi-psi spatial edges
                r_edges = frozenset(
                    (operators[chosen_phis[perm[j]]].spatial_arg, psi_spatial[j])
                    for j in range(n_psi)
                )
                cached = _r_cycle_cache.get(r_edges)
                if cached is None:
                    # Quick inline cycle check on just these R edges
                    adj: dict[str, list[str]] = {}
                    for sl, sr in r_edges:
                        adj.setdefault(sl, []).append(sr)
                    WHITE, GRAY, BLACK = 0, 1, 2
                    colour: dict[str, int] = {}
                    for v in adj:
                        colour[v] = WHITE
                    for targets in adj.values():
                        for t in targets:
                            colour.setdefault(t, WHITE)

                    def _dfs_early(node: str) -> bool:
                        colour[node] = GRAY
                        for nb in adj.get(node, ()):
                            if colour[nb] == GRAY:
                                return True
                            if colour[nb] == WHITE and _dfs_early(nb):
                                return True
                        colour[node] = BLACK
                        return False

                    cached = any(
                        _dfs_early(v) for v, c in list(colour.items()) if c == WHITE
                    )
                    _r_cycle_cache[r_edges] = cached
                if cached:
                    continue

            r_pairs = tuple(
                (chosen_phis[perm[j]], psi_indices[j]) for j in range(n_psi)
            )

            # All ways to pair remaining phi's among themselves
            for c_pairing in generate_all_pairings(remaining_phis):
                yield r_pairs + c_pairing


def _has_r_cycle(propagators: list[Propagator]) -> bool:
    """Check whether the R propagators form a directed cycle.

    A cycle R(a₁,a₂) R(a₂,a₃) ... R(aₙ,a₁) vanishes by causality
    because R is retarded: it would require t₁ > t₂ > ... > tₙ > t₁.
    """
    # Build directed adjacency from R spatial args (skip equal-point,
    # which is already handled by the Itô rule).
    adj: dict[str, list[str]] = {}
    for p in propagators:
        if p.kind == "R" and p.spatial_left != p.spatial_right:
            adj.setdefault(p.spatial_left, []).append(p.spatial_right)

    if not adj:
        return False

    # Standard DFS cycle detection (white/gray/black colouring)
    WHITE, GRAY, BLACK = 0, 1, 2
    colour: dict[str, int] = {v: WHITE for v in adj}
    # Also register nodes that only appear as targets
    for targets in adj.values():
        for t in targets:
            colour.setdefault(t, WHITE)

    def _dfs(node: str) -> bool:
        colour[node] = GRAY
        for nb in adj.get(node, ()):
            if colour[nb] == GRAY:
                return True  # back edge → cycle
            if colour[nb] == WHITE and _dfs(nb):
                return True
        colour[node] = BLACK
        return False

    return any(
        _dfs(v) for v, c in list(colour.items()) if c == WHITE
    )


def evaluate_pairing(
    operators: Sequence[FieldOperator],
    pairing: Pairing,
    ito: bool = True,
) -> Optional[tuple[Expr, list[Propagator]]]:
    r"""Evaluate a single complete pairing.

    Returns ``(Product of propagators, list of individual propagators)``
    or ``None`` if any contraction vanishes.

    Vanishing conditions:

    - :math:`\psi`\ --\ :math:`\psi` contraction
    - Equal-point :math:`R(x,x)=0` when *ito* is ``True``
    - Causal R-loop: any directed cycle among R propagator spatial
      arguments (e.g. :math:`R(a,b)\,R(b,a)=0`) when *ito* is ``True``

    Args:
        operators: The full list of field operators.
        pairing: Tuple of ``(i, j)`` index pairs.
        ito: If ``True``, apply the Itô prescription and causal
            vanishing rules.
    """
    propagators: list[Propagator] = []
    for i, j in pairing:
        prop = contract_pair(operators[i], operators[j], ito=ito)
        if prop is None:
            return None
        propagators.append(prop)

    if len(propagators) == 0:
        return None

    # Causal vanishing: R-loop detection (with memoization by spatial edges)
    if ito:
        r_spatial_key = frozenset(
            (p.spatial_left, p.spatial_right)
            for p in propagators
            if p.kind == "R" and p.spatial_left != p.spatial_right
        )
        if r_spatial_key:
            cached = _r_cycle_cache.get(r_spatial_key)
            if cached is None:
                cached = _has_r_cycle(propagators)
                _r_cycle_cache[r_spatial_key] = cached
            if cached:
                return None

    if len(propagators) == 1:
        return propagators[0], propagators

    return Product(tuple(propagators)), propagators


def wick_contract(
    operators: Sequence[FieldOperator],
    ito: bool = True,
) -> tuple[Expr, list[Pairing]]:
    r"""Apply Wick's theorem to a product of field operators.

    Args:
        operators: Sequence of field operators to contract.
        ito: If ``True``, apply the Itô prescription
            :math:`\Theta(0)=0`: response propagators at equal spatial
            points vanish, i.e. :math:`R(x,x)=0`.

    Returns:
        ``(expression, surviving_pairings)`` where *expression* is a
        Sum of Products of Propagators, and *surviving_pairings* lists
        the non-zero pairings.
    """
    _r_cycle_cache.clear()

    n = len(operators)
    if n == 0:
        return ZERO, []
    if n % 2 != 0:
        return ZERO, []

    # Separate phi and psi operators (by index in the operators list)
    phi_indices: list[int] = []
    psi_indices: list[int] = []
    for idx, op in enumerate(operators):
        if op.field_type == FieldType.PHYSICAL:
            phi_indices.append(idx)
        else:
            psi_indices.append(idx)

    # Check feasibility
    if len(psi_indices) > len(phi_indices):
        return ZERO, []
    if (len(phi_indices) - len(psi_indices)) % 2 != 0:
        return ZERO, []

    terms: list[Expr] = []
    surviving_pairings: list[Pairing] = []

    for pairing in generate_valid_pairings(
        phi_indices, psi_indices, operators=operators, ito=ito
    ):
        result = evaluate_pairing(operators, pairing, ito=ito)
        if result is not None:
            expr, _props = result
            terms.append(expr)
            surviving_pairings.append(pairing)

    if not terms:
        return ZERO, []
    if len(terms) == 1:
        return terms[0], surviving_pairings

    return Sum(tuple(terms)), surviving_pairings


# Spatial signature: sorted tuple of canonical (kind, spatial_left, spatial_right)
SpatialSignature = tuple[tuple[str, str, str], ...]


def _canonical_edge(kind: str, sl: str, sr: str) -> tuple[str, str, str]:
    """Canonical form for a propagator edge (C is symmetric)."""
    if kind == "C":
        return ("C", min(sl, sr), max(sl, sr))
    return (kind, sl, sr)


def wick_contract_grouped(
    operators: Sequence[FieldOperator],
    ito: bool = True,
) -> tuple[
    dict[SpatialSignature, list[tuple[list[Propagator], Pairing]]],
    list[Pairing],
]:
    r"""Apply Wick's theorem, grouping results by spatial propagator signature.

    Like :func:`wick_contract` but instead of returning a flat Sum of
    Products, groups surviving pairings by their spatial propagator
    topology.  Pairings in the same group share the same Feynman diagram
    but differ in component-index routing.

    Returns:
        ``(groups, all_pairings)`` where *groups* maps each spatial
        signature to a list of ``(propagator_list, pairing)`` tuples,
        and *all_pairings* is the flat list of surviving pairings.
    """
    _r_cycle_cache.clear()

    n = len(operators)
    if n == 0:
        return {}, []
    if n % 2 != 0:
        return {}, []

    phi_indices: list[int] = []
    psi_indices: list[int] = []
    for idx, op in enumerate(operators):
        if op.field_type == FieldType.PHYSICAL:
            phi_indices.append(idx)
        else:
            psi_indices.append(idx)

    if len(psi_indices) > len(phi_indices):
        return {}, []
    if (len(phi_indices) - len(psi_indices)) % 2 != 0:
        return {}, []

    groups: dict[SpatialSignature, list[tuple[list[Propagator], Pairing]]] = {}
    all_pairings: list[Pairing] = []

    for pairing in generate_valid_pairings(
        phi_indices, psi_indices, operators=operators, ito=ito
    ):
        result = evaluate_pairing(operators, pairing, ito=ito)
        if result is None:
            continue
        _expr, props = result
        all_pairings.append(pairing)

        sig = tuple(sorted(
            _canonical_edge(p.kind, p.spatial_left, p.spatial_right)
            for p in props
        ))
        groups.setdefault(sig, []).append((props, pairing))

    return groups, all_pairings


# ---------------------------------------------------------------------------
# Spatial-level Wick contraction engine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpatialTopology:
    """A spatial-level Wick pairing.

    Represents a complete contraction at the spatial-point level,
    abstracting away component indices.
    """

    r_edges: tuple[tuple[str, str], ...]  # sorted (phi_point, psi_point)
    c_edges: tuple[tuple[str, str], ...]  # sorted canonical (min, max) pairs
    multiplicity: int  # equivalent operator-level pairings


def _has_r_cycle_edges(r_edges: list[tuple[str, str]]) -> bool:
    """Check whether directed R-edges form a cycle."""
    adj: dict[str, list[str]] = {}
    for sl, sr in r_edges:
        if sl != sr:
            adj.setdefault(sl, []).append(sr)
    if not adj:
        return False
    WHITE, GRAY, BLACK = 0, 1, 2
    colour: dict[str, int] = {}
    for v in adj:
        colour[v] = WHITE
    for targets in adj.values():
        for t in targets:
            colour.setdefault(t, WHITE)

    def _dfs(node: str) -> bool:
        colour[node] = GRAY
        for nb in adj.get(node, ()):
            if colour[nb] == GRAY:
                return True
            if colour[nb] == WHITE and _dfs(nb):
                return True
        colour[node] = BLACK
        return False

    return any(_dfs(v) for v, c in list(colour.items()) if c == WHITE)


def _enumerate_r_assignments(
    psi_list: list[str],
    phi_capacity: dict[str, int],
    ito: bool,
    idx: int = 0,
    r_so_far: list[tuple[str, str]] | None = None,
) -> Iterator[list[tuple[str, str]]]:
    """Enumerate valid R-assignments at the spatial level.

    Each ψ at point ``psi_list[idx]`` is assigned to a φ at some point
    with remaining capacity.

    Deduplication: consecutive ψ's at the same point get non-decreasing
    φ-source ordering.
    """
    if r_so_far is None:
        r_so_far = []

    if idx == len(psi_list):
        if not _has_r_cycle_edges(r_so_far):
            yield list(r_so_far)
        return

    psi_point = psi_list[idx]

    # Determine minimum φ-source for dedup of same-point ψ's
    min_phi: str | None = None
    if idx > 0 and psi_list[idx] == psi_list[idx - 1]:
        min_phi = r_so_far[-1][0]

    for phi_point in sorted(phi_capacity.keys()):
        if phi_capacity[phi_point] <= 0:
            continue
        if ito and phi_point == psi_point:
            continue
        if min_phi is not None and phi_point < min_phi:
            continue

        phi_capacity[phi_point] -= 1
        r_so_far.append((phi_point, psi_point))

        yield from _enumerate_r_assignments(
            psi_list, phi_capacity, ito, idx + 1, r_so_far
        )

        r_so_far.pop()
        phi_capacity[phi_point] += 1


def _enumerate_c_pairings(
    remaining: dict[str, int],
    _min_cross_partner: str | None = None,
) -> Iterator[list[tuple[str, str]]]:
    """Enumerate ways to pair remaining φ-slots into C propagators.

    Always picks from the lexicographically first available point to
    ensure each spatial C-pairing is yielded exactly once.

    When a point has multiple remaining slots going to cross-edges,
    ``_min_cross_partner`` enforces non-decreasing partner ordering
    to avoid generating the same spatial pairing twice.
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

    # Self-loop (no ordering constraint — self-loops are distinct from cross)
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


def _compute_multiplicity(
    c_edges: list[tuple[str, str]],
    vertex_phi_counts: dict[str, int],
) -> int:
    r"""Compute the operator-level multiplicity for a spatial topology.

    Formula: :math:`\frac{\prod_v m_v!}{2^{N_{\text{self}}} \cdot \prod_e k_e!}`

    where *m_v* is the number of φ operators from vertex *v*,
    *N_self* is the total number of C self-loops, and
    *k_e* is the number of parallel C edges between each pair of
    spatial points (parallel edges are indistinguishable).

    ``vertex_phi_counts`` maps each **vertex** spatial point to its φ count.
    Observable points are excluded (they have multiplicity 1).
    """
    # Count self-loops per spatial point
    self_loops: dict[str, int] = {}
    for s1, s2 in c_edges:
        if s1 == s2:
            self_loops[s1] = self_loops.get(s1, 0) + 1

    mult = 1
    for v_point, m_v in vertex_phi_counts.items():
        n_self = self_loops.get(v_point, 0)
        mult *= factorial(m_v) // (2 ** n_self)

    # Divide by k! for each group of k parallel C edges
    edge_counts: dict[tuple[str, str], int] = {}
    for s1, s2 in c_edges:
        key = (min(s1, s2), max(s1, s2))
        edge_counts[key] = edge_counts.get(key, 0) + 1
    for count in edge_counts.values():
        if count > 1:
            mult //= factorial(count)

    return mult


def wick_contract_spatial(
    operators: Sequence[FieldOperator],
    ito: bool = True,
    vertex_points: frozenset[str] | None = None,
) -> dict[SpatialSignature, tuple[list[Propagator], int, Pairing]]:
    r"""Spatial-level Wick contraction.

    Enumerates spatial topologies (not individual operator-level pairings)
    and computes a multiplicity for each.  This avoids the combinatorial
    explosion from component-index routing.

    Args:
        operators: Sequence of field operators.
        ito: Apply the Itô prescription.
        vertex_points: Frozenset of spatial args that belong to vertices
            (not observables).  Needed for multiplicity computation.

    Returns:
        Dict mapping each :data:`SpatialSignature` to
        ``(reference_propagators, multiplicity, representative_pairing)``.
    """
    _r_cycle_cache.clear()

    n = len(operators)
    if n == 0 or n % 2 != 0:
        return {}

    # Group operators by (spatial_arg, field_type)
    phi_at: dict[str, list[int]] = {}  # spatial -> [operator indices]
    psi_at: dict[str, list[int]] = {}
    for idx, op in enumerate(operators):
        if op.field_type == FieldType.PHYSICAL:
            phi_at.setdefault(op.spatial_arg, []).append(idx)
        else:
            psi_at.setdefault(op.spatial_arg, []).append(idx)

    phi_counts = {s: len(ops) for s, ops in phi_at.items()}
    psi_counts = {s: len(ops) for s, ops in psi_at.items()}

    total_phi = sum(phi_counts.values())
    total_psi = sum(psi_counts.values())
    if total_psi > total_phi:
        return {}
    if (total_phi - total_psi) % 2 != 0:
        return {}

    # Build flat ψ-list for R-assignment enumeration (sorted for determinism)
    psi_list: list[str] = []
    for s in sorted(psi_counts.keys()):
        psi_list.extend([s] * psi_counts[s])

    # Vertex φ-counts for multiplicity (exclude observable points)
    if vertex_points is None:
        vertex_points = frozenset()
    vertex_phi_counts = {
        s: c for s, c in phi_counts.items() if s in vertex_points
    }

    results: dict[SpatialSignature, tuple[list[Propagator], int, Pairing]] = {}

    phi_cap = dict(phi_counts)

    for r_assign in _enumerate_r_assignments(psi_list, phi_cap, ito):
        # Compute remaining φ-capacity after R-assignment
        remaining = dict(phi_counts)
        for phi_s, _psi_s in r_assign:
            remaining[phi_s] -= 1

        for c_pairing in _enumerate_c_pairings(remaining):
            # Compute multiplicity
            mult = _compute_multiplicity(c_pairing, vertex_phi_counts)

            # Build canonical spatial signature
            sig_edges: list[tuple[str, str, str]] = []
            for phi_s, psi_s in r_assign:
                sig_edges.append(("R", phi_s, psi_s))
            for s1, s2 in c_pairing:
                sig_edges.append(_canonical_edge("C", s1, s2))
            sig: SpatialSignature = tuple(sorted(sig_edges))

            if sig in results:
                # Same signature reached via different enumeration path
                # (shouldn't happen with canonical enumeration, but be safe)
                old_props, old_mult, old_pairing = results[sig]
                results[sig] = (old_props, old_mult + mult, old_pairing)
                continue

            # Build representative propagators and pairing
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
                phi_op = operators[phi_idx]
                psi_op = operators[psi_idx]
                props.append(Propagator(
                    kind="R",
                    index_left=phi_op.component_index,
                    index_right=psi_op.component_index,
                    spatial_left=phi_op.spatial_arg,
                    spatial_right=psi_op.spatial_arg,
                ))
                pairs.append((phi_idx, psi_idx))

            for s1, s2 in c_pairing:
                idx1 = phi_at[s1][phi_used[s1]]
                phi_used[s1] += 1
                idx2 = phi_at[s2][phi_used[s2]]
                phi_used[s2] += 1
                op1 = operators[idx1]
                op2 = operators[idx2]
                props.append(Propagator(
                    kind="C",
                    index_left=op1.component_index,
                    index_right=op2.component_index,
                    spatial_left=op1.spatial_arg,
                    spatial_right=op2.spatial_arg,
                ))
                pairs.append((idx1, idx2))

            results[sig] = (props, mult, tuple(pairs))

    return results
