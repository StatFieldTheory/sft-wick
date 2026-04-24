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

from collections import defaultdict
from fractions import Fraction
from itertools import permutations

from .expressions import (
    ZERO,
    ONE,
    Expr,
    IntegralOver,
    KroneckerDelta,
    Product,
    Propagator,
    Rational,
    Sum,
    SumOverIndex,
    Symbol,
)


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


def _apply_perm_to_symbol(sym: Symbol, perm: dict[str, str]) -> Symbol:
    """Relabel indices of a Symbol according to the permutation."""
    new_indices = tuple(perm.get(idx, idx) for idx in sym.indices)
    new_spatial = tuple(perm.get(s, s) for s in sym.spatial_args)
    if new_indices == sym.indices and new_spatial == sym.spatial_args:
        return sym
    return Symbol(sym.name, new_indices, new_spatial)


def _apply_perm_to_expr(expr: Expr, perm: dict[str, str]) -> Expr:
    """Apply index permutation to Symbols and Rationals in an expression."""
    if not perm:
        return expr
    if isinstance(expr, Symbol):
        return _apply_perm_to_symbol(expr, perm)
    if isinstance(expr, Rational):
        return expr
    if isinstance(expr, Product):
        return Product(tuple(_apply_perm_to_expr(f, perm) for f in expr.factors))
    return expr


def _canonical_edge(
    kind: str, spatial_left: str, spatial_right: str,
) -> tuple[str, str, str]:
    """Canonical form for a single propagator edge.

    C edges are symmetric: ``C(x, y) = C(y, x)``, so we sort the
    spatial arguments.  R edges are directed and kept as-is.
    """
    if kind == "C":
        return ("C", min(spatial_left, spatial_right),
                max(spatial_left, spatial_right))
    return (kind, spatial_left, spatial_right)


def _canonical_diagram_form(
    props: list[Propagator],
    integration_vars: frozenset[str],
) -> tuple[tuple[tuple[str, str, str], ...], dict[str, str]]:
    """Compute canonical graph form for a set of propagators.

    Tries all permutations of integration variables (internal spatial
    points), keeping external spatial variables fixed.  For each
    permutation, computes the sorted edge list using
    :func:`_canonical_edge`, then picks the lexicographically smallest.

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
            _canonical_edge(p.kind, p.spatial_left, p.spatial_right)
            for p in props
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
            edges_list.append(_canonical_edge(p.kind, sl, sr))
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
        key = _canonical_edge(p.kind, p.spatial_left, p.spatial_right)
        ref_groups[key].append(i)

    # Group remapped other props similarly, tracking whether C was flipped
    other_groups: dict[tuple[str, str, str], list[tuple[int, bool]]] = defaultdict(list)
    for i, (p, sl, sr, _) in enumerate(remapped):
        canon_key = _canonical_edge(p.kind, sl, sr)
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
            ok = True
            for ri, oi, flipped in option:
                rp = ref_props[ri]
                op = other_props[oi]

                if flipped:
                    pairs = [(rp.index_left, op.index_right),
                             (rp.index_right, op.index_left)]
                else:
                    pairs = [(rp.index_left, op.index_left),
                             (rp.index_right, op.index_right)]

                for ref_idx, other_idx in pairs:
                    if ref_idx is None and other_idx is None:
                        continue
                    if ref_idx == other_idx:
                        continue
                    if (other_idx not in internal_indices
                            or ref_idx not in internal_indices):
                        ok = False
                        break
                    if other_idx in new_perm:
                        if new_perm[other_idx] != ref_idx:
                            ok = False
                            break
                    else:
                        new_perm[other_idx] = ref_idx
                if not ok:
                    break

            if ok:
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
            _canonical_edge(p.kind, p.spatial_left, p.spatial_right)
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
            # Single term — reconstruct with coupling symbols included
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
                # Cannot merge — add as separate term with coupling
                parts = list(outer_symbols) + non_props + props
                ungroupable.append(
                    Product(tuple(parts)) if len(parts) > 1 else parts[0]
                )
                continue

            # Full permutation
            full_perm = {**spatial_perm, **comp_perm}

            # Apply to outer coupling symbols + inner non-props
            permuted_factors: list[Expr] = []
            for sym in outer_symbols:
                permuted_factors.append(_apply_perm_to_expr(sym, full_perm))
            for np in non_props:
                permuted_factors.append(_apply_perm_to_expr(np, full_perm))

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
        Simplified expression with the requested constraints applied.
    """
    diag_R = diag_R or iso_R
    diag_C = diag_C or iso_C
    if not diag_R and not diag_C:
        return expr
    result = _diag_walk(expr, diag_R, diag_C, iso_R, iso_C, {})
    result = simplify(result)
    return _canonical_expr_indices(result)


def _diag_walk(
    expr: Expr,
    diag_R: bool,
    diag_C: bool,
    iso_R: bool,
    iso_C: bool,
    sum_dims: dict[str, int],
) -> Expr:
    """Recursively walk expression, applying diagonal propagator constraints.

    *sum_dims* maps summation-index names to their dimension, accumulated
    from ancestor :class:`SumOverIndex` wrappers.
    """
    if isinstance(expr, SumOverIndex):
        new_sum_dims = {**sum_dims, expr.index_name: expr.dimension}
        new_body = _diag_walk(expr.body, diag_R, diag_C, iso_R, iso_C, new_sum_dims)
        if not _expr_uses_index(new_body, expr.index_name):
            return new_body
        return SumOverIndex(expr.index_name, expr.dimension, new_body)

    if isinstance(expr, IntegralOver):
        new_body = _diag_walk(expr.body, diag_R, diag_C, iso_R, iso_C, sum_dims)
        return IntegralOver(expr.variable, new_body)

    if isinstance(expr, Sum):
        new_terms = [_diag_walk(t, diag_R, diag_C, iso_R, iso_C, sum_dims) for t in expr.terms]
        new_terms = [t for t in new_terms if not _is_zero(t)]
        if not new_terms:
            return ZERO
        if len(new_terms) == 1:
            return new_terms[0]
        return Sum(tuple(new_terms))

    if isinstance(expr, Product):
        has_props = any(isinstance(f, Propagator) for f in expr.factors)
        if has_props:
            return _diag_product(expr.factors, diag_R, diag_C, iso_R, iso_C, sum_dims)
        new_factors = [
            _diag_walk(f, diag_R, diag_C, iso_R, iso_C, sum_dims) for f in expr.factors
        ]
        return Product(tuple(new_factors))

    if isinstance(expr, Propagator):
        # Bare propagator (not inside a Product) — wrap and apply
        if (
            expr.index_left is not None
            and expr.index_right is not None
        ):
            if expr.index_left != expr.index_right:
                if (expr.kind == "R" and diag_R) or (expr.kind == "C" and diag_C):
                    return _diag_product((expr,), diag_R, diag_C, iso_R, iso_C, sum_dims)
            elif expr.index_left == expr.index_right:
                if (expr.kind == "R" and iso_R) or (expr.kind == "C" and iso_C):
                    return Propagator(expr.kind, None, None, expr.spatial_left, expr.spatial_right)
        return expr

    return expr


def _diag_product(
    factors: tuple[Expr, ...],
    diag_R: bool,
    diag_C: bool,
    iso_R: bool,
    iso_C: bool,
    sum_dims: dict[str, int],
) -> Expr:
    """Apply diagonal (and optionally isotropic) constraints to a Product that contains Propagators."""
    sum_idx_set = set(sum_dims.keys())

    constraints: list[tuple[str, str]] = []
    for f in factors:
        if (
            isinstance(f, Propagator)
            and f.index_left is not None
            and f.index_right is not None
        ):
            if (f.kind == "R" and diag_R) or (f.kind == "C" and diag_C):
                if f.index_left != f.index_right:
                    constraints.append((f.index_left, f.index_right))

    if not constraints:
        return Product(tuple(factors)) if len(factors) > 1 else factors[0]

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
        if rr in sum_idx_set and rl not in sum_idx_set:
            parent[rr] = rl
        elif rl in sum_idx_set and rr not in sum_idx_set:
            parent[rl] = rr
        else:
            parent[rr] = rl

    sub: dict[str, str] = {}
    for idx in {k for pair in constraints for k in pair}:
        root = find(idx)
        if idx != root:
            sub[idx] = root

    if not sub:
        return Product(tuple(factors)) if len(factors) > 1 else factors[0]

    new_factors = [_apply_index_sub(f, sub) for f in factors]

    # Add KroneckerDelta for external-external constraints
    seen_deltas: set[frozenset[str]] = set()
    for idx, root in sub.items():
        if idx not in sum_idx_set and root not in sum_idx_set:
            pair = frozenset({idx, root})
            if pair not in seen_deltas:
                seen_deltas.add(pair)
                a, b = sorted(pair)
                new_factors.append(KroneckerDelta(a, b))

    # Isotropic: strip equal component indices from R/C propagators
    if iso_R or iso_C:
        new_factors = [
            Propagator(f.kind, None, None, f.spatial_left, f.spatial_right)
            if (
                isinstance(f, Propagator)
                and f.index_left is not None
                and f.index_left == f.index_right
                and ((f.kind == "R" and iso_R) or (f.kind == "C" and iso_C))
            )
            else f
            for f in new_factors
        ]

    if len(new_factors) == 1:
        return new_factors[0]
    return Product(tuple(new_factors))


def _canonical_expr_indices(expr: Expr) -> Expr:
    """Rename SumOverIndex variables to i_0, i_1, ... in DFS first-appearance order."""
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
    rename = {old: f"i_{j}" for j, old in enumerate(seen) if old != f"i_{j}"}
    if not rename:
        return expr
    return _apply_index_sub(expr, rename)


def _apply_index_sub(expr: Expr, sub: dict[str, str]) -> Expr:
    """Apply an index-name substitution to an expression."""
    if not sub:
        return expr
    if isinstance(expr, Propagator):
        il = sub.get(expr.index_left, expr.index_left) if expr.index_left else expr.index_left
        ir = sub.get(expr.index_right, expr.index_right) if expr.index_right else expr.index_right
        if il == expr.index_left and ir == expr.index_right:
            return expr
        return Propagator(expr.kind, il, ir, expr.spatial_left, expr.spatial_right)
    if isinstance(expr, Symbol):
        new_indices = tuple(sub.get(i, i) for i in expr.indices)
        new_spatial = tuple(sub.get(s, s) for s in expr.spatial_args)
        if new_indices == expr.indices and new_spatial == expr.spatial_args:
            return expr
        return Symbol(expr.name, new_indices, new_spatial)
    if isinstance(expr, Sum):
        return Sum(tuple(_apply_index_sub(t, sub) for t in expr.terms))
    if isinstance(expr, Product):
        return Product(tuple(_apply_index_sub(f, sub) for f in expr.factors))
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


def _multiply_into(expr: Expr, factor: int) -> Expr:
    """Multiply *expr* by an integer, pushing the factor into inner Products.

    Descends through :class:`SumOverIndex`, :class:`IntegralOver`, and
    :class:`Sum` wrappers so that the factor lands next to the existing
    :class:`Rational` coefficient, enabling :func:`simplify` to absorb
    it in a single pass.
    """
    if isinstance(expr, Rational):
        return Rational(factor * expr.numerator, expr.denominator)
    if isinstance(expr, Product):
        return Product((Rational(factor, 1),) + expr.factors)
    if isinstance(expr, SumOverIndex):
        return SumOverIndex(
            expr.index_name, expr.dimension, _multiply_into(expr.body, factor),
        )
    if isinstance(expr, IntegralOver):
        return IntegralOver(expr.variable, _multiply_into(expr.body, factor))
    if isinstance(expr, Sum):
        return Sum(tuple(_multiply_into(t, factor) for t in expr.terms))
    return Product((Rational(factor, 1), expr))


def _expr_uses_index(expr: Expr, name: str) -> bool:
    """Check whether *expr* references index *name* anywhere."""
    if isinstance(expr, Propagator):
        return name == expr.index_left or name == expr.index_right
    if isinstance(expr, Symbol):
        return name in expr.indices
    if isinstance(expr, Product):
        return any(_expr_uses_index(f, name) for f in expr.factors)
    if isinstance(expr, Sum):
        return any(_expr_uses_index(t, name) for t in expr.terms)
    if isinstance(expr, SumOverIndex):
        if expr.index_name == name:
            return False  # shadowed by inner sum
        return _expr_uses_index(expr.body, name)
    if isinstance(expr, IntegralOver):
        return _expr_uses_index(expr.body, name)
    if isinstance(expr, KroneckerDelta):
        return name == expr.index1 or name == expr.index2
    return False
