"""Expression simplification for Wick contraction results.

Pipeline:
1. Flatten nested Sum/Product
2. Absorb rational prefactors in products
3. Eliminate zeros
4. Canonical ordering of propagators
5. Term collection (combine like terms)
6. Diagram-based collection (group by Feynman diagram isomorphism)
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Collection, Iterable
from fractions import Fraction
from itertools import permutations, product
from typing import NamedTuple

from .expressions import (
    ZERO,
    ONE,
    DiracDelta,
    Expr,
    IntegralOver,
    KroneckerDelta,
    Product,
    Propagator,
    Rational,
    Sum,
    SumOverIndex,
    Symbol,
    canonical_edge,
)
from .indices import fresh_names

#: Most additive terms :func:`diagonal_propagators` expands one product into.
_MAX_DIAGONAL_TERMS = 2**20


def simplify(expr: Expr) -> Expr:
    """Main simplification entry point."""
    expr = _flatten(expr)
    expr = _absorb_rationals(expr)
    expr = _eliminate_zeros(expr)
    expr = _collect_terms(expr)
    return expr


def _flatten(expr: Expr) -> Expr:
    """Recursively flatten nested Sum and Product."""
    if isinstance(expr, Sum):
        flat_terms: list[Expr] = []
        for t in expr.terms:
            t = _flatten(t)
            if isinstance(t, Sum):
                flat_terms.extend(t.terms)
            else:
                flat_terms.append(t)
        return Sum(tuple(flat_terms))

    if isinstance(expr, Product):
        flat_factors: list[Expr] = []
        for f in expr.factors:
            f = _flatten(f)
            if isinstance(f, Product):
                flat_factors.extend(f.factors)
            else:
                flat_factors.append(f)
        return Product(tuple(flat_factors))

    if isinstance(expr, IntegralOver):
        return IntegralOver(expr.variable, _flatten(expr.body))

    if isinstance(expr, SumOverIndex):
        return SumOverIndex(expr.index_name, expr.dimension, _flatten(expr.body))

    return expr


def _absorb_rationals(expr: Expr) -> Expr:
    """In a Product, multiply all Rational factors into a single prefactor."""
    if isinstance(expr, Product):
        coeff = Fraction(1)
        other_factors: list[Expr] = []
        for f in expr.factors:
            f = _absorb_rationals(f)
            if isinstance(f, Rational):
                coeff *= f.to_fraction()
            else:
                other_factors.append(f)

        if coeff == 0:
            return ZERO
        if not other_factors:
            return Rational(coeff.numerator, coeff.denominator)
        if coeff == 1:
            if len(other_factors) == 1:
                return other_factors[0]
            return Product(tuple(other_factors))
        return Product(
            (Rational(coeff.numerator, coeff.denominator), *other_factors)
        )

    if isinstance(expr, Sum):
        return Sum(tuple(_absorb_rationals(t) for t in expr.terms))

    if isinstance(expr, IntegralOver):
        return IntegralOver(expr.variable, _absorb_rationals(expr.body))

    if isinstance(expr, SumOverIndex):
        return SumOverIndex(
            expr.index_name, expr.dimension, _absorb_rationals(expr.body)
        )

    return expr


def _eliminate_zeros(expr: Expr) -> Expr:
    """Remove zero terms from sums; collapse products containing zero."""
    if isinstance(expr, Sum):
        terms = [_eliminate_zeros(t) for t in expr.terms]
        terms = [t for t in terms if not _is_zero(t)]
        if not terms:
            return ZERO
        if len(terms) == 1:
            return terms[0]
        return Sum(tuple(terms))

    if isinstance(expr, Product):
        factors = [_eliminate_zeros(f) for f in expr.factors]
        if any(_is_zero(f) for f in factors):
            return ZERO
        factors = [f for f in factors if not _is_one(f)]
        if not factors:
            return ONE
        if len(factors) == 1:
            return factors[0]
        return Product(tuple(factors))

    if isinstance(expr, IntegralOver):
        body = _eliminate_zeros(expr.body)
        if _is_zero(body):
            return ZERO
        return IntegralOver(expr.variable, body)

    if isinstance(expr, SumOverIndex):
        body = _eliminate_zeros(expr.body)
        if _is_zero(body):
            return ZERO
        return SumOverIndex(expr.index_name, expr.dimension, body)

    return expr


def _is_zero(expr: Expr) -> bool:
    return isinstance(expr, Rational) and expr.is_zero


def _is_one(expr: Expr) -> bool:
    return isinstance(expr, Rational) and expr.is_one


def _propagator_key(p: Propagator) -> tuple:
    """Canonical sort key for a propagator."""
    return (p.kind, p.spatial_left, p.spatial_right, p.index_left or "", p.index_right or "")


def _term_signature(expr: Expr) -> tuple | None:
    """Extract the non-coefficient part of a term for grouping.

    A term is either:
      - A single Propagator
      - A Product of (optional Rational) * Propagators * Symbols * IntegralOvers * SumOverIndices
    Returns a hashable signature, or None if it can't be grouped.
    """
    if isinstance(expr, Propagator):
        return (("prop", _propagator_key(expr)),)

    if isinstance(expr, Product):
        parts: list[tuple] = []
        for f in expr.factors:
            if isinstance(f, Rational):
                continue  # skip coefficient
            if isinstance(f, Propagator):
                parts.append(("prop", _propagator_key(f)))
            elif isinstance(f, Symbol):
                parts.append(("sym", f.name, f.indices, f.spatial_args))
            elif isinstance(f, IntegralOver):
                parts.append(("int", f.variable))
            elif isinstance(f, SumOverIndex):
                parts.append(("sum", f.index_name, f.dimension))
            else:
                return None  # can't group
        return tuple(sorted(parts))

    return None


def _get_coefficient(expr: Expr) -> Fraction:
    """Extract the rational coefficient from a term."""
    if isinstance(expr, Rational):
        return expr.to_fraction()
    if isinstance(expr, Product):
        coeff = Fraction(1)
        for f in expr.factors:
            if isinstance(f, Rational):
                coeff *= f.to_fraction()
        return coeff
    return Fraction(1)


def _set_coefficient(expr: Expr, coeff: Fraction) -> Expr:
    """Replace the rational coefficient of a term."""
    if isinstance(expr, Rational):
        return Rational(coeff.numerator, coeff.denominator)

    if isinstance(expr, Product):
        non_rational = [f for f in expr.factors if not isinstance(f, Rational)]
        if coeff == 1:
            if len(non_rational) == 1:
                return non_rational[0]
            return Product(tuple(non_rational))
        return Product(
            (Rational(coeff.numerator, coeff.denominator), *non_rational)
        )

    # expr has implicit coefficient 1
    if coeff == 1:
        return expr
    return Product((Rational(coeff.numerator, coeff.denominator), expr))


def _collect_terms(expr: Expr) -> Expr:
    """Combine terms with identical propagator structures."""
    if not isinstance(expr, Sum):
        return expr

    groups: defaultdict[tuple | None, list[Expr]] = defaultdict(list)
    ungroupable: list[Expr] = []

    for term in expr.terms:
        sig = _term_signature(term)
        if sig is not None:
            groups[sig].append(term)
        else:
            ungroupable.append(term)

    collected: list[Expr] = []
    for sig, terms in groups.items():
        total_coeff = sum((_get_coefficient(t) for t in terms), Fraction(0))
        if total_coeff != 0:
            representative = terms[0]
            collected.append(_set_coefficient(representative, total_coeff))

    collected.extend(ungroupable)

    if not collected:
        return ZERO
    if len(collected) == 1:
        return collected[0]
    return Sum(tuple(collected))


# ---------------------------------------------------------------------------
# Diagram-based collection
# ---------------------------------------------------------------------------


def _complete_permutation(perm: dict[str, str]) -> dict[str, str] | None:
    """Extend a renaming of dummy labels to a permutation of them.

    Matching propagators names only the labels they carry.  A label that
    lives in the coupling alone is still a dummy, and it has to move out of
    the way: ``{j: i}`` alone would turn ``F_ij`` into ``F_ii``.

    Returns ``None`` when two labels map to one, which no renaming of dummies
    does.
    """
    images = set(perm.values())
    if len(images) != len(perm):
        return None
    return {
        **perm,
        **dict(zip(sorted(images - perm.keys()), sorted(perm.keys() - images))),
    }


def _same_time_edges(
    same_time_groups: tuple[tuple[str, ...], ...],
    mapping: dict[str, str],
) -> list[tuple[str, str, str]]:
    """``("E", u, v)`` for every pair of legs in one equal-time group,
    after relabeling by ``mapping``."""
    edges: list[tuple[str, str, str]] = []
    for group in same_time_groups:
        mapped = sorted(mapping.get(v, v) for v in group)
        for i, u in enumerate(mapped):
            for v in mapped[i + 1:]:
                edges.append(("E", u, v))
    return edges


def _canonical_diagram_form(
    props: list[Propagator],
    integration_vars: frozenset[str],
    same_time_groups: tuple[tuple[str, ...], ...] = (),
) -> tuple[tuple[tuple[str, str, str], ...], dict[str, str]]:
    """Compute canonical graph form for a set of propagators.

    Tries all permutations of integration variables (internal spatial
    points), keeping external spatial variables fixed.  For each
    permutation, computes the sorted edge list using
    :func:`~sft_wick.expressions.canonical_edge`, then picks the lexicographically smallest.

    ``same_time_groups`` lists the leg labels of each ``equal_time``
    non-local vertex instance: the legs of one group share one time
    variable.  Each group enters the form as a clique of ``("E", u, v)``
    edges, so two contractions have the same form only if a relabeling
    maps every group onto a group.  The propagators alone do not record
    which legs share a time: two copies of an equal-time vertex whose
    legs meet six external points as {x1,x2,x3}{x4,x5,x6} and as
    {x1,x2,x4}{x3,x5,x6} have isomorphic propagator graphs and different
    time integrals.

    Returns ``(canonical_edges, best_mapping)`` where *best_mapping*
    maps original internal spatial variable names to the permuted names
    that produced the winning form.
    """
    # Collect all spatial variables appearing in propagators
    all_spatial: set[str] = set()
    for p in props:
        all_spatial.add(p.spatial_left)
        all_spatial.add(p.spatial_right)

    # Internal = integration variables that actually appear in propagators
    internal = sorted(all_spatial & integration_vars)

    # Trivial case: no internal variables to permute
    if not internal:
        form = tuple(sorted(
            [canonical_edge(p.kind, p.spatial_left, p.spatial_right)
             for p in props] + _same_time_edges(same_time_groups, {})
        ))
        return form, {}

    best_form: tuple[tuple[str, str, str], ...] | None = None
    best_mapping: dict[str, str] = {}

    # Try all permutations of internal variables (k! where k = |internal| ≤ 4
    # in practice, so at most 24 iterations).  Signature-based pruning is
    # deliberately avoided: local 1-hop signatures only capture graph
    # automorphisms, not the cross-graph relabelings needed for canonical
    # form minimization, and can incorrectly prevent the true minimum from
    # being found (e.g. when two nodes have different local degrees but the
    # graph is still isomorphic to another via a cross-signature permutation).
    for perm in permutations(internal):
        mapping: dict[str, str] = dict(zip(internal, perm))
        edges_list: list[tuple[str, str, str]] = []
        for p in props:
            sl = mapping.get(p.spatial_left, p.spatial_left)
            sr = mapping.get(p.spatial_right, p.spatial_right)
            edges_list.append(canonical_edge(p.kind, sl, sr))
        edges_list.extend(_same_time_edges(same_time_groups, mapping))
        form = tuple(sorted(edges_list))
        if best_form is None or form < best_form:
            best_form = form
            best_mapping = mapping

    assert best_form is not None
    return best_form, best_mapping


def _canonical_key_nauty(
    props: list[Propagator],
    integration_vars: frozenset[str],
) -> bytes:
    """Compute a canonical graph certificate using nauty (via pynauty).

    Returns a ``bytes`` object that uniquely identifies the graph up
    to permutation of internal (integration) spatial variables.  Two
    propagator lists produce the same certificate if and only if they
    represent isomorphic Feynman diagrams.

    The diagram is encoded as a directed vertex-colored graph with
    dummy intermediate nodes for edge-type encoding:

    - **R(a, b)**: ``a → R_dummy → b`` (two directed edges)
    - **C(a, b)**: ``a ↔ C_dummy ↔ b`` (bidirectional through dummy)

    Vertex coloring fixes external nodes (each gets a unique color)
    while allowing internal nodes and same-type dummies to be permuted.

    Falls back to :func:`_canonical_diagram_form` if ``pynauty`` is
    not installed.
    """
    try:
        import pynauty
    except ImportError:
        canon, _mapping = _canonical_diagram_form(props, integration_vars)
        # Return bytes for consistency
        return str(canon).encode()

    # Collect spatial points
    all_spatial: set[str] = set()
    for p in props:
        all_spatial.add(p.spatial_left)
        all_spatial.add(p.spatial_right)

    external = sorted(all_spatial - integration_vars)
    internal = sorted(all_spatial & integration_vars)

    node_id: dict[str, int] = {}
    for i, name in enumerate(external):
        node_id[name] = i
    for i, name in enumerate(internal):
        node_id[name] = len(external) + i

    n_real = len(node_id)
    r_list = [p for p in props if p.kind == "R"]
    c_list = [p for p in props if p.kind == "C"]
    r_start = n_real
    c_start = n_real + len(r_list)
    total = n_real + len(r_list) + len(c_list)

    adj: dict[int, set[int]] = defaultdict(set)

    for k, p in enumerate(r_list):
        dummy = r_start + k
        left, right = node_id[p.spatial_left], node_id[p.spatial_right]
        adj[left].add(dummy)
        adj[dummy].add(right)

    for k, p in enumerate(c_list):
        dummy = c_start + k
        left, right = node_id[p.spatial_left], node_id[p.spatial_right]
        adj[left].add(dummy)
        adj[dummy].add(left)
        adj[right].add(dummy)
        adj[dummy].add(right)

    # Vertex coloring: each external is unique, internals share,
    # R-dummies share, C-dummies share
    coloring: list[set[int]] = []
    for i in range(len(external)):
        coloring.append({i})
    if internal:
        coloring.append(set(range(len(external), n_real)))
    if r_list:
        coloring.append(set(range(r_start, c_start)))
    if c_list:
        coloring.append(set(range(c_start, total)))

    g = pynauty.Graph(total, directed=True, vertex_coloring=coloring)
    for v, nbrs in adj.items():
        if nbrs:
            g.connect_vertex(v, sorted(nbrs))

    return pynauty.certificate(g)


def _try_add_index_perm(
    perm: dict[str, str],
    ref_prop: Propagator,
    other_prop: Propagator,
    flipped: bool,
    internal_indices: set[str],
) -> bool:
    """Try to add component-index mappings from other_prop to ref_prop."""
    if flipped:
        pairs = [
            (ref_prop.index_left, other_prop.index_right),
            (ref_prop.index_right, other_prop.index_left),
        ]
    else:
        pairs = [
            (ref_prop.index_left, other_prop.index_left),
            (ref_prop.index_right, other_prop.index_right),
        ]
    for ref_idx, other_idx in pairs:
        if ref_idx is None and other_idx is None:
            continue
        if ref_idx == other_idx:
            continue
        if other_idx not in internal_indices or ref_idx not in internal_indices:
            return False
        if other_idx in perm:
            if perm[other_idx] != ref_idx:
                return False
        else:
            perm[other_idx] = ref_idx
    return True


def _match_propagators_after_spatial(
    ref_props: list[Propagator],
    other_props: list[Propagator],
    spatial_perm: dict[str, str],
    internal_indices: set[str],
) -> dict[str, str] | None:
    """Find the component-index permutation after spatial relabeling.

    Applies *spatial_perm* to *other_props*' spatial arguments, then
    matches them to *ref_props* by ``(kind, spatial_left,
    spatial_right)``.  For C propagators that match with reversed
    spatial arguments (C symmetry), the component indices are swapped.

    When multiple propagators share the same spatial signature, all
    valid matchings are tried to find one that yields a consistent
    component-index permutation.

    Returns a dict ``{other_index -> ref_index}`` for internal
    indices, or ``None`` if no consistent matching exists.
    """
    # Apply spatial perm to other props
    remapped: list[tuple[Propagator, str, str, bool]] = []
    for p in other_props:
        sl = spatial_perm.get(p.spatial_left, p.spatial_left)
        sr = spatial_perm.get(p.spatial_right, p.spatial_right)
        remapped.append((p, sl, sr, False))

    # Group ref props by (kind, canonical_spatial_left, canonical_spatial_right)
    # where canonical means sorted for C
    ref_groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for i, p in enumerate(ref_props):
        key = canonical_edge(p.kind, p.spatial_left, p.spatial_right)
        ref_groups[key].append(i)

    # Group remapped other props similarly, tracking whether C was flipped
    other_groups: dict[tuple[str, str, str], list[tuple[int, bool]]] = defaultdict(list)
    for i, (p, sl, sr, _) in enumerate(remapped):
        canon_key = canonical_edge(p.kind, sl, sr)
        flipped = (p.kind == "C" and sl > sr)
        other_groups[canon_key].append((i, flipped))

    # Check that groups match
    if set(ref_groups.keys()) != set(other_groups.keys()):
        return None
    for key in ref_groups:
        if len(ref_groups[key]) != len(other_groups[key]):
            return None

    # For each group, try all orderings of the other props
    # Build the full matching as a Cartesian product of per-group permutations
    group_keys = sorted(ref_groups.keys())
    group_matchings: list[list[list[tuple[int, int, bool]]]] = []

    for key in group_keys:
        ri_list = ref_groups[key]
        oi_list = other_groups[key]
        # Try all permutations of other indices against ref indices
        options: list[list[tuple[int, int, bool]]] = []
        for oi_perm in permutations(oi_list):
            matching = []
            for ri, (oi, flipped) in zip(ri_list, oi_perm):
                # Also check if ref C needs flipping
                rp = ref_props[ri]
                _, osl, osr, _ = remapped[oi]
                if rp.kind == "C":
                    both_canonical = (
                        min(rp.spatial_left, rp.spatial_right) == min(osl, osr)
                        and max(rp.spatial_left, rp.spatial_right) == max(osl, osr)
                    )
                    if not both_canonical:
                        continue
                    # Determine if the actual (non-canonical) orders differ
                    actual_flipped = (rp.spatial_left, rp.spatial_right) != (osl, osr)
                    matching.append((ri, oi, actual_flipped))
                else:
                    # R: must match exactly after spatial perm
                    if (rp.spatial_left, rp.spatial_right) != (osl, osr):
                        continue
                    matching.append((ri, oi, False))
            if len(matching) == len(ri_list):
                options.append(matching)
        if not options:
            return None
        group_matchings.append(options)

    # Try all combinations of per-group matchings
    def _try_combinations(
        group_idx: int,
        perm: dict[str, str],
    ) -> dict[str, str] | None:
        if group_idx == len(group_matchings):
            return perm

        for option in group_matchings[group_idx]:
            new_perm = dict(perm)
            if all(
                _try_add_index_perm(
                    new_perm, ref_props[ri], other_props[oi], flipped,
                    internal_indices,
                )
                for ri, oi, flipped in option
            ):
                result = _try_combinations(group_idx + 1, new_perm)
                if result is not None:
                    return result

        return None

    return _try_combinations(0, {})


# ---------------------------------------------------------------------------
# Public API: collect_by_diagram
# ---------------------------------------------------------------------------


def collect_by_diagram(expr: Expr) -> Expr:
    """Collect terms whose Feynman diagrams are isomorphic.

    Groups terms that represent the same Feynman diagram under
    relabeling of dummy integration variables and summation indices.
    Factors out canonical propagators and sums coupling coefficients
    with appropriately permuted indices.

    This correctly handles:

    - **Spatial variable relabeling**: integration variables can be
      freely permuted (e.g. ``y_0 ↔ y_1`` at second order).
    - **C propagator symmetry**: ``C(x, y) = C(y, x)``.
    - **Coupling permutation**: index permutations are applied to the
      outer coupling symbols, producing a sum of permuted couplings.

    Example::

        F_{i0 i1 i2} × [R_{a i2}(x,y) C_{i0 i1}(y,y)
                         + R_{a i1}(x,y) C_{i0 i2}(y,y)]

    becomes::

        (F_{i0 i1 i2} + F_{i0 i2 i1}) × R_{a i2}(x,y) C_{i0 i1}(y,y)
    """
    return _collect_inner(expr, frozenset(), frozenset())


def _collect_inner(
    expr: Expr,
    integration_vars: frozenset[str],
    summation_indices: frozenset[str],
) -> Expr:
    """Recursive worker for :func:`collect_by_diagram`.

    Accumulates integration variables and summation indices as it
    descends through wrapper nodes, then processes the inner
    Product/Sum structure.
    """
    if isinstance(expr, IntegralOver):
        new_ivars = integration_vars | {expr.variable}
        body = _collect_inner(expr.body, new_ivars, summation_indices)
        return IntegralOver(expr.variable, body)

    if isinstance(expr, SumOverIndex):
        new_sindices = summation_indices | {expr.index_name}
        body = _collect_inner(expr.body, integration_vars, new_sindices)
        return SumOverIndex(expr.index_name, expr.dimension, body)

    if isinstance(expr, Sum):
        return Sum(tuple(
            _collect_inner(t, integration_vars, summation_indices)
            for t in expr.terms
        ))

    if not isinstance(expr, Product):
        return expr

    # --- Product: look for an inner Sum to collect ---
    sum_idx = None
    for i, f in enumerate(expr.factors):
        if isinstance(f, Sum):
            sum_idx = i
            break
    if sum_idx is None:
        return expr

    the_sum = expr.factors[sum_idx]
    outer_factors = list(expr.factors[:sum_idx]) + list(expr.factors[sum_idx + 1:])

    # Separate outer factors by type
    outer_symbols: list[Symbol] = []
    outer_other: list[Expr] = []
    for f in outer_factors:
        if isinstance(f, Symbol):
            outer_symbols.append(f)
        else:
            outer_other.append(f)

    # Internal indices: those that appear in coupling symbols
    internal_indices: set[str] = set()
    for sym in outer_symbols:
        internal_indices.update(sym.indices)

    if not internal_indices and not integration_vars:
        return expr  # nothing to collect

    # --- Group Sum terms by canonical diagram form ---
    # Pre-group by exact spatial signature to avoid redundant canonical
    # form computations.  Terms with identical spatial signatures share
    # the same canonical form, so _canonical_diagram_form is called once
    # per spatial group instead of once per term.
    groups: dict[
        tuple[tuple[str, str, str], ...],
        list[tuple[list[Propagator], list[Expr], dict[str, str]]],
    ] = {}
    ungroupable: list[Expr] = []

    spatial_sig_groups: dict[
        tuple[tuple[str, str, str], ...],
        list[tuple[list[Propagator], list[Expr]]],
    ] = {}

    for term in the_sum.terms:
        if isinstance(term, Product):
            props = [f for f in term.factors if isinstance(f, Propagator)]
            non_props = [f for f in term.factors if not isinstance(f, Propagator)]
        elif isinstance(term, Propagator):
            props = [term]
            non_props = []
        else:
            ungroupable.append(term)
            continue

        if not props:
            ungroupable.append(term)
            continue

        sig = tuple(sorted(
            canonical_edge(p.kind, p.spatial_left, p.spatial_right)
            for p in props
        ))
        spatial_sig_groups.setdefault(sig, []).append((props, non_props))

    for sig, entries in spatial_sig_groups.items():
        # Compute canonical form once for the group representative
        ref_props = entries[0][0]
        canonical_form, mapping = _canonical_diagram_form(ref_props, integration_vars)
        for props, non_props in entries:
            groups.setdefault(canonical_form, []).append((props, non_props, mapping))

    # --- Merge each group ---
    new_sum_terms: list[Expr] = []

    for canonical_form, group in groups.items():
        if len(group) == 1:
            # Single term: reconstruct with coupling symbols included
            props, non_props, _ = group[0]
            parts: list[Expr] = list(outer_symbols) + non_props + props
            if len(parts) == 1:
                new_sum_terms.append(parts[0])
            else:
                new_sum_terms.append(Product(tuple(parts)))
            continue

        # Multiple terms with same canonical diagram form
        ref_props, ref_non_props, ref_mapping = group[0]
        ref_inv = {v: k for k, v in ref_mapping.items()}

        coupling_terms: list[Expr] = []

        for props, non_props, mapping in group:
            # Spatial permutation: other -> canonical -> ref
            spatial_perm: dict[str, str] = {}
            for orig, canon in mapping.items():
                target = ref_inv.get(canon, canon)
                if orig != target:
                    spatial_perm[orig] = target

            # Component index permutation via propagator matching
            comp_perm = _match_propagators_after_spatial(
                ref_props, props, spatial_perm, internal_indices,
            )
            if comp_perm is None:
                # Cannot merge: add as separate term with coupling
                parts = list(outer_symbols) + non_props + props
                ungroupable.append(
                    Product(tuple(parts)) if len(parts) > 1 else parts[0]
                )
                continue

            full_perm = _complete_permutation({**spatial_perm, **comp_perm})
            if full_perm is None:
                parts = list(outer_symbols) + non_props + props
                ungroupable.append(
                    Product(tuple(parts)) if len(parts) > 1 else parts[0]
                )
                continue

            # Apply to outer coupling symbols + inner non-props
            permuted_factors = [
                _apply_index_sub(f, full_perm) for f in (*outer_symbols, *non_props)
            ]

            if not permuted_factors:
                coupling_terms.append(ONE)
            elif len(permuted_factors) == 1:
                coupling_terms.append(permuted_factors[0])
            else:
                coupling_terms.append(Product(tuple(permuted_factors)))

        if not coupling_terms:
            continue

        # Build coupling sum
        if len(coupling_terms) == 1:
            coupling_expr: Expr = coupling_terms[0]
        else:
            # Fast path: sum bare Rationals directly
            if all(isinstance(t, Rational) for t in coupling_terms):
                total = sum(t.to_fraction() for t in coupling_terms)
                coupling_expr = Rational(total.numerator, total.denominator)
            else:
                coupling_expr = simplify(Sum(tuple(coupling_terms)))

        collected_term = Product((coupling_expr,) + tuple(ref_props))
        new_sum_terms.append(collected_term)

    new_sum_terms.extend(ungroupable)

    if not new_sum_terms:
        return Product(tuple(outer_other)) if outer_other else ZERO

    if len(new_sum_terms) == 1:
        new_inner = new_sum_terms[0]
    else:
        new_inner = Sum(tuple(new_sum_terms))

    if outer_other:
        return Product(tuple(outer_other) + (new_inner,))
    return new_inner


# Keep old name as alias for backward compatibility
collect_by_topology = collect_by_diagram


# ---------------------------------------------------------------------------
# Diagonal propagator simplification
# ---------------------------------------------------------------------------


def diagonal_propagators(
    expr: Expr,
    *,
    diag_R: bool = False,
    diag_C: bool = False,
    iso_R: bool = False,
    iso_C: bool = False,
) -> Expr:
    r"""Enforce diagonal and/or isotropic propagator constraints.

    When ``diag_R=True``, response propagators are diagonal:
    :math:`R_{ij}(x,y) = \delta_{ij}\,R(x,y)`.  This substitutes the
    constraint :math:`i = j` into couplings and propagators, eliminating
    one summation index per constrained propagator.

    When ``diag_C=True``, the same is done for correlation propagators.

    When ``iso_R=True`` (implies ``diag_R=True``), all diagonal R entries
    are further assumed equal: :math:`R_{ii}(x,y) = R(x,y)`.  The
    remaining component index is dropped from R propagators entirely.

    When ``iso_C=True`` (implies ``diag_C=True``), the same for C.

    Args:
        expr: Expression to simplify.
        diag_R: Enforce diagonal response propagators.
        diag_C: Enforce diagonal correlation propagators.
        iso_R: Enforce isotropic (index-free) response propagators.
        iso_C: Enforce isotropic (index-free) correlation propagators.

    Returns:
        Simplified expression with the requested constraints applied.  A
        product of sums comes back expanded into additive terms.

    Raises:
        ValueError: if a summed index or integrated point is bound twice in
            one product, a sum rebinds a name inside it, or a product would
            expand past ``2**20`` terms.
    """
    diag_R = diag_R or iso_R
    diag_C = diag_C or iso_C
    if not diag_R and not diag_C:
        return expr
    reduced = [
        _diag_term(term, diag_R, diag_C, iso_R, iso_C)
        for term in _additive_terms(expr)
    ]
    return _canonical_expr_indices(simplify(_rebuild_terms(reduced)))


def diagonal_index_sub(
    propagators: Iterable[Propagator],
    summed: Collection[str],
    diag_R: bool,
    diag_C: bool,
) -> tuple[dict[str, str], list[KroneckerDelta]]:
    r"""Index substitution implied by the diagonal propagators of one term.

    Args:
        propagators: The propagators of a single additive term.
        summed: The indices that term sums over.
        diag_R: Enforce diagonal response propagators.
        diag_C: Enforce diagonal correlation propagators.

    Returns:
        ``(sub, deltas)``.  ``sub`` maps each merged index to the survivor
        of its class, an external index when the class has one, since
        :math:`\sum_i \delta_{ia} f(i) = f(a)`.  A summed key of ``sub`` is
        eliminated from this term only.  ``deltas`` holds one
        :class:`KroneckerDelta` for each merged pair of external indices,
        which no sum absorbs.
    """
    constraints = [
        (p.index_left, p.index_right)
        for p in propagators
        if p.index_left is not None
        and p.index_right is not None
        and p.index_left != p.index_right
        and ((p.kind == "R" and diag_R) or (p.kind == "C" and diag_C))
    ]

    # Union-find for index equivalences (prefer external index as root)
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while x in parent:
            x = parent[x]
        return x

    for left, right in constraints:
        rl, rr = find(left), find(right)
        if rl == rr:
            continue
        if rl in summed and rr not in summed:
            parent[rl] = rr
        else:
            parent[rr] = rl

    sub = {
        idx: find(idx)
        for idx in sorted({k for pair in constraints for k in pair})
        if find(idx) != idx
    }
    pairs = {
        tuple(sorted((idx, root)))
        for idx, root in sub.items()
        if idx not in summed and root not in summed
    }
    return sub, [KroneckerDelta(a, b) for a, b in sorted(pairs)]


def strip_isotropic(p: Propagator, iso_R: bool, iso_C: bool) -> Propagator:
    """Drop the equal component indices of ``p`` under an isotropy flag."""
    if (
        p.index_left is not None
        and p.index_left == p.index_right
        and ((p.kind == "R" and iso_R) or (p.kind == "C" and iso_C))
    ):
        return Propagator(p.kind, None, None, p.spatial_left, p.spatial_right)
    return p


class _DiagTerm(NamedTuple):
    """One additive term: its own sums and integrals over a flat product.

    A delta constraint belongs to a single term, so the sums have to be
    held per term: in sum_i (G_i C_ai + H_ii C_bb) the first
    term loses i and the second keeps it.
    """

    sums: tuple[tuple[str, int], ...]
    integrals: tuple[str, ...]
    factors: tuple[Expr, ...]

    def bound(self) -> set[str]:
        return {name for name, _ in self.sums} | set(self.integrals)

    def labels(self) -> set[str]:
        out = self.bound()
        for f in self.factors:
            out |= expr_labels(f)
        return out


def _has_structure(expr: Expr) -> bool:
    """True when ``expr`` holds a propagator, a sum or an integral."""
    if isinstance(expr, (Propagator, SumOverIndex, IntegralOver)):
        return True
    if isinstance(expr, Sum):
        return any(_has_structure(t) for t in expr.terms)
    if isinstance(expr, Product):
        return any(_has_structure(f) for f in expr.factors)
    return False


def _refuse_rebinding(names: set[str]) -> None:
    """Pulling a sum or integral outwards must not capture another label."""
    if names:
        raise ValueError(
            "diagonal_propagators needs each summed index and integrated "
            f"point to be bound once in a product; got {sorted(names)}"
        )


def _multiply_terms(left: _DiagTerm, right: _DiagTerm) -> _DiagTerm:
    """Product of two terms, with their sums and integrals pulled out."""
    _refuse_rebinding(
        (left.bound() & right.labels()) | (right.bound() & left.labels())
    )
    return _DiagTerm(
        left.sums + right.sums,
        left.integrals + right.integrals,
        left.factors + right.factors,
    )


def _additive_terms(expr: Expr) -> list[_DiagTerm]:
    """Expand ``expr`` into additive terms.

    A factor with no propagator, sum or integral (a coupling sum, say) stays
    whole.
    """
    if isinstance(expr, (SumOverIndex, IntegralOver)):
        outer = _binder(expr)
        terms = _additive_terms(expr.body)
        for term in terms:
            _refuse_rebinding(term.bound() & outer.bound())
        return [
            _DiagTerm(
                outer.sums + t.sums, outer.integrals + t.integrals, t.factors
            )
            for t in terms
        ]
    if isinstance(expr, Sum) and _has_structure(expr):
        return [term for t in expr.terms for term in _additive_terms(t)]
    if isinstance(expr, Product) and _has_structure(expr):
        parts = [_additive_terms(f) for f in expr.factors]
        size = math.prod(len(part) for part in parts)
        if size > _MAX_DIAGONAL_TERMS:
            raise ValueError(
                f"diagonal_propagators would expand a product into {size} "
                f"terms, above the limit of {_MAX_DIAGONAL_TERMS}."
            )
        terms = [_DiagTerm((), (), ())]
        for part in parts:
            terms = [
                _multiply_terms(left, right) for left, right in product(terms, part)
            ]
        return terms
    return [_DiagTerm((), (), (expr,))]


def _binder(expr: SumOverIndex | IntegralOver) -> _DiagTerm:
    """The empty term carrying the sum or integral that ``expr`` opens."""
    if isinstance(expr, SumOverIndex):
        return _DiagTerm(((expr.index_name, expr.dimension),), (), ())
    return _DiagTerm((), (expr.variable,), ())


def _diag_term(
    term: _DiagTerm, diag_R: bool, diag_C: bool, iso_R: bool, iso_C: bool
) -> _DiagTerm:
    """Apply the constraints of one term, removing only the sums it eliminates."""
    summed = {name for name, _ in term.sums}
    propagators = [f for f in term.factors if isinstance(f, Propagator)]
    sub, deltas = diagonal_index_sub(propagators, summed, diag_R, diag_C)
    factors = [_apply_index_sub(f, sub) for f in term.factors]
    factors = [
        strip_isotropic(f, iso_R, iso_C) if isinstance(f, Propagator) else f
        for f in factors
    ]
    return _DiagTerm(
        tuple(s for s in term.sums if s[0] not in sub),
        term.integrals,
        (*factors, *deltas),
    )


def _rebuild_terms(terms: list[_DiagTerm]) -> Expr:
    """Sum the terms, sharing the sums and integrals of those that agree."""
    groups: dict[tuple, list[Expr]] = {}
    for term in terms:
        body = (
            term.factors[0]
            if len(term.factors) == 1
            else Product(term.factors) if term.factors else ONE
        )
        groups.setdefault((term.sums, term.integrals), []).append(body)

    out: list[Expr] = []
    for (sums, integrals), bodies in groups.items():
        expr = bodies[0] if len(bodies) == 1 else Sum(tuple(bodies))
        for variable in reversed(integrals):
            expr = IntegralOver(variable, expr)
        for name, dimension in reversed(sums):
            expr = SumOverIndex(name, dimension, expr)
        out.append(expr)
    if not out:
        return ZERO
    return out[0] if len(out) == 1 else Sum(tuple(out))


def _canonical_expr_indices(expr: Expr) -> Expr:
    """Rename SumOverIndex variables to i_0, i_1, ... in DFS first-appearance order.

    The new names skip every other label in ``expr`` (an observable index
    or point spelled ``i_0``, say), which would otherwise be captured by
    the renamed sum.
    """
    seen: list[str] = []
    seen_set: set[str] = set()

    def _collect(e: Expr) -> None:
        if isinstance(e, SumOverIndex):
            if e.index_name not in seen_set:
                seen.append(e.index_name)
                seen_set.add(e.index_name)
            _collect(e.body)
        elif isinstance(e, IntegralOver):
            _collect(e.body)
        elif isinstance(e, Product):
            for child in e.factors:
                _collect(child)
        elif isinstance(e, Sum):
            for child in e.terms:
                _collect(child)

    _collect(expr)
    targets = fresh_names("i", len(seen), expr_labels(expr) - seen_set)
    rename = {old: new for old, new in zip(seen, targets) if old != new}
    if not rename:
        return expr
    return _apply_index_sub(expr, rename)


def expr_labels(expr: Expr) -> set[str]:
    """Every component-index name and spatial label in ``expr``, bound or
    free: propagator and symbol indices and points, Kronecker and Dirac
    delta arguments, summed indices and integrated points."""
    out: set[str] = set()

    def _walk(e: Expr) -> None:
        if isinstance(e, Propagator):
            out.update(
                n
                for n in (e.index_left, e.index_right, e.spatial_left, e.spatial_right)
                if n is not None
            )
        elif isinstance(e, Symbol):
            out.update(e.indices)
            out.update(e.spatial_args)
        elif isinstance(e, KroneckerDelta):
            out.update((e.index1, e.index2))
        elif isinstance(e, DiracDelta):
            out.update((e.arg1, e.arg2))
        elif isinstance(e, SumOverIndex):
            out.add(e.index_name)
            _walk(e.body)
        elif isinstance(e, IntegralOver):
            out.add(e.variable)
            _walk(e.body)
        elif isinstance(e, Product):
            for child in e.factors:
                _walk(child)
        elif isinstance(e, Sum):
            for child in e.terms:
                _walk(child)

    _walk(expr)
    return out


def _apply_index_sub(expr: Expr, sub: dict[str, str]) -> Expr:
    """Rename labels in an expression, the one relabeller of the package.

    A symbol's indices and points, a propagator's indices, delta indices
    and summed indices are renamed.  A propagator's points and an integrated
    point are not: the callers that move points do that themselves.
    """
    if not sub:
        return expr
    if isinstance(expr, Symbol):
        new_indices = tuple(sub.get(i, i) for i in expr.indices)
        new_spatial = tuple(sub.get(s, s) for s in expr.spatial_args)
        if new_indices == expr.indices and new_spatial == expr.spatial_args:
            return expr
        # ``local`` suppresses the single point of a local coupling when
        # rendering.
        return Symbol(expr.name, new_indices, new_spatial, expr.local)
    if isinstance(expr, Product):
        return Product(tuple(_apply_index_sub(f, sub) for f in expr.factors))
    if isinstance(expr, Sum):
        return Sum(tuple(_apply_index_sub(t, sub) for t in expr.terms))
    if isinstance(expr, Propagator):
        il = sub.get(expr.index_left, expr.index_left) if expr.index_left else expr.index_left
        ir = sub.get(expr.index_right, expr.index_right) if expr.index_right else expr.index_right
        if il == expr.index_left and ir == expr.index_right:
            return expr
        return Propagator(expr.kind, il, ir, expr.spatial_left, expr.spatial_right)
    if isinstance(expr, KroneckerDelta):
        i1 = sub.get(expr.index1, expr.index1)
        i2 = sub.get(expr.index2, expr.index2)
        if i1 == expr.index1 and i2 == expr.index2:
            return expr
        return KroneckerDelta(i1, i2)
    if isinstance(expr, SumOverIndex):
        new_name = sub.get(expr.index_name, expr.index_name)
        new_body = _apply_index_sub(expr.body, sub)
        if new_name == expr.index_name and new_body is expr.body:
            return expr
        return SumOverIndex(new_name, expr.dimension, new_body)
    if isinstance(expr, IntegralOver):
        new_body = _apply_index_sub(expr.body, sub)
        if new_body is expr.body:
            return expr
        return IntegralOver(expr.variable, new_body)
    return expr
