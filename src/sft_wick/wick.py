"""Core Wick contraction engine.

Enumerates all valid pairings (contractions) of field operators and evaluates
each pairing as a product of propagators.
"""

from __future__ import annotations

from itertools import combinations, permutations
from typing import Iterator, Optional, Sequence

from .expressions import Expr, Product, Propagator, Sum, ZERO
from .fields import FieldOperator, FieldType
from .propagators import contract_pair

# A pairing is a tuple of (index_i, index_j) pairs
Pairing = tuple[tuple[int, int], ...]


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
) -> Iterator[Pairing]:
    """Generate only non-vanishing pairings (no psi-psi contractions).

    Exploits the MSR structure: each psi must pair with a phi (producing R),
    and remaining phi's pair among themselves (producing C).

    This avoids generating the many zero-valued psi-psi pairings.
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

    # Choose which phi's pair with the psi's
    for phi_subset_indices in combinations(range(n_phi), n_psi):
        chosen_phis = [phi_indices[k] for k in phi_subset_indices]
        remaining_phis = [
            phi_indices[k] for k in range(n_phi) if k not in phi_subset_indices
        ]

        # All ways to match psi's to chosen phi's
        for perm in permutations(range(n_psi)):
            r_pairs = tuple(
                (chosen_phis[perm[j]], psi_indices[j]) for j in range(n_psi)
            )

            # All ways to pair remaining phi's among themselves
            for c_pairing in generate_all_pairings(remaining_phis):
                yield r_pairs + c_pairing


def evaluate_pairing(
    operators: Sequence[FieldOperator],
    pairing: Pairing,
) -> Optional[tuple[Expr, list[Propagator]]]:
    """Evaluate a single complete pairing.

    Returns (Product of propagators, list of individual propagators) or None
    if any contraction vanishes (psi-psi).
    """
    propagators: list[Propagator] = []
    for i, j in pairing:
        prop = contract_pair(operators[i], operators[j])
        if prop is None:
            return None
        propagators.append(prop)

    if len(propagators) == 0:
        return None

    if len(propagators) == 1:
        return propagators[0], propagators

    return Product(tuple(propagators)), propagators


def wick_contract(
    operators: Sequence[FieldOperator],
) -> tuple[Expr, list[Pairing]]:
    """Apply Wick's theorem to a product of field operators.

    Returns:
        (expression, surviving_pairings) where expression is a Sum of Products
        of Propagators, and surviving_pairings lists the non-zero pairings.
    """
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

    for pairing in generate_valid_pairings(phi_indices, psi_indices):
        result = evaluate_pairing(operators, pairing)
        if result is not None:
            expr, _props = result
            terms.append(expr)
            surviving_pairings.append(pairing)

    if not terms:
        return ZERO, []
    if len(terms) == 1:
        return terms[0], surviving_pairings

    return Sum(tuple(terms)), surviving_pairings
