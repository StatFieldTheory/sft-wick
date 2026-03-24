"""Perturbative expansion driver.

Computes <O>_S = sum_{n=0}^{N} (-1)^n / n! * <O * S_int^n>_{S_0}
using Wick's theorem for each order.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import TYPE_CHECKING

from .action import Action
from .expressions import (
    ZERO,
    Expr,
    IntegralOver,
    Product,
    Propagator,
    Rational,
    Sum,
    SumOverIndex,
    Symbol,
    apply_response_phase,
)
from .fields import FieldOperator, FieldType
from .indices import IndexContext
from .propagators import contract_pair
from .simplify import (
    _apply_index_sub,
    _canonical_diagram_form,
    _match_propagators_after_spatial,
    collect_by_diagram,
    diagonal_propagators,
    simplify,
)
from .vertices import VertexInstance
from .wick import Pairing, SpatialSignature, wick_contract, wick_contract_spatial

if TYPE_CHECKING:
    pass


class PerturbativeResult:
    """Container for the result of a perturbative calculation.

    Stores the symbolic expression at each perturbative order together
    with the corresponding Feynman diagrams.

    Attributes:
        order_terms: Mapping from perturbative order *n* to the
            simplified expression for that order.
        total: Sum of all non-zero order contributions.
        diagrams_by_order: Mapping from perturbative order *n* to the
            list of :class:`DiagramInfo` records at that order.
    """

    def __init__(
        self,
        order_terms: dict[int, Expr],
        total: Expr,
        diagrams_by_order: dict[int, list[DiagramInfo]],
        diagram_terms_by_order: dict[int, list[DiagramTerm]] | None = None,
    ) -> None:
        self.order_terms = order_terms
        self.total = total
        self.diagrams_by_order = diagrams_by_order
        self.diagram_terms_by_order: dict[int, list[DiagramTerm]] = (
            diagram_terms_by_order or {}
        )

    def order(self, n: int) -> Expr:
        """Get the contribution at a specific perturbative order.

        Args:
            n: The perturbative order (0, 1, 2, ...).

        Returns:
            The simplified expression at order *n*, or ``ZERO`` if that
            order has no contribution.
        """
        return self.order_terms.get(n, ZERO)

    def diagram_terms(self, order: int) -> list[DiagramTerm]:
        """Structured diagram contributions for numerical evaluation.

        Each :class:`DiagramTerm` carries the propagators, coupling
        coefficient, prefactor, and index structure needed to evaluate
        a single Feynman diagram numerically.

        Populated when ``collect_topology=True`` (the default).

        Args:
            order: The perturbative order.

        Returns:
            List of :class:`DiagramTerm` at that order, or empty list.
        """
        return self.diagram_terms_by_order.get(order, [])

    def to_latex(self) -> str:
        """Generate LaTeX for the full result, order by order.

        Returns:
            A multi-line string with one ``O(n): <latex>`` line per
            non-zero order.
        """
        parts: list[str] = []
        for n in sorted(self.order_terms.keys()):
            expr = self.order_terms[n]
            if not _is_zero(expr):
                parts.append(f"O({n}): {expr.to_latex()}")
        return "\n".join(parts)

    def draw_diagrams(self, order: int | None = None, **kwargs) -> None:
        """Draw Feynman diagrams using matplotlib.

        Topologically identical diagrams are drawn only once, with a
        ``×N`` multiplicity label when *N* > 1.

        Args:
            order: If given, draw only diagrams at this perturbative
                order.  Otherwise draw all diagrams.
            **kwargs: Forwarded to :class:`~sft_wick.drawing.DiagramRenderer`
                (e.g. ``figsize``).
        """
        from collections import OrderedDict

        from .diagrams import FeynmanDiagram
        from .drawing import DiagramRenderer

        renderer = DiagramRenderer(**kwargs)
        if order is not None:
            diagrams = self.diagrams_by_order.get(order, [])
        else:
            diagrams = []
            for o in sorted(self.diagrams_by_order.keys()):
                diagrams.extend(self.diagrams_by_order[o])

        if not diagrams:
            print("No diagrams to draw.")
            return

        # Build FeynmanDiagram objects and deduplicate by canonical form
        fd_list = [d.to_feynman_diagram() for d in diagrams]

        groups: OrderedDict[tuple, list[FeynmanDiagram]] = OrderedDict()
        for fd in fd_list:
            key = fd.canonical_form()
            if key not in groups:
                groups[key] = []
            groups[key].append(fd)

        unique_diagrams = [group[0] for group in groups.values()]
        multiplicities = [len(group) for group in groups.values()]

        renderer.draw_all(unique_diagrams, multiplicities=multiplicities)

    def __repr__(self) -> str:
        return self.to_latex()


class DiagramInfo:
    """Lightweight record of a Feynman diagram for deferred construction.

    The actual :class:`~sft_wick.diagrams.FeynmanDiagram` graph is built
    lazily via :meth:`to_feynman_diagram` to avoid up-front cost when
    many diagrams are generated.

    Attributes:
        observable_ops: Field operators forming the observable.
        vertex_instances: Instantiated vertices contributing to this diagram.
        pairing: The Wick contraction pairing (tuple of index pairs).
        coefficient: Rational prefactor for this diagram.
        order: The perturbative order at which this diagram appears.
    """

    def __init__(
        self,
        observable_ops: list[FieldOperator],
        vertex_instances: list[VertexInstance],
        pairing: Pairing,
        coefficient: Rational,
        order: int,
    ) -> None:
        self.observable_ops = observable_ops
        self.vertex_instances = vertex_instances
        self.pairing = pairing
        self.coefficient = coefficient
        self.order = order

    def to_feynman_diagram(self):
        """Construct a :class:`~sft_wick.diagrams.FeynmanDiagram` from this record.

        Returns:
            A fully-constructed ``FeynmanDiagram`` graph with external
            nodes, vertices, and propagator edges.
        """
        from .diagrams import FeynmanDiagram

        return FeynmanDiagram.from_pairing(
            self.observable_ops,
            self.vertex_instances,
            self.pairing,
        )


@dataclass(frozen=True)
class DiagramTerm:
    """A single Feynman diagram's contribution, structured for numerical evaluation.

    The full contribution is::

        rational_prefactor × response_phase_factor × coupling_sum × ∏ propagators

    summed over ``summation_indices`` and integrated over ``integration_vars``.

    Attributes:
        propagators: Tuple of propagators forming the diagram.
        coupling_sum: Symbolic coupling expression (sum of permuted
            couplings), **without** the rational prefactor.
        rational_prefactor: The ``(-1)^n / n! × multinomial`` coefficient.
        integration_vars: Spatial variables to integrate over.
        summation_indices: ``(index_name, dimension)`` pairs for component
            index summations.
        n_response: Number of R propagators (determines the response phase).
    """

    propagators: tuple[Propagator, ...]
    coupling_sum: Expr
    rational_prefactor: Rational
    integration_vars: tuple[str, ...]
    summation_indices: tuple[tuple[str, int], ...]
    n_response: int

    def spatial_topology(self) -> list[tuple[str, str, str]]:
        """Return ``(kind, spatial_left, spatial_right)`` for each propagator."""
        return [
            (p.kind, p.spatial_left, p.spatial_right) for p in self.propagators
        ]

    def response_phase_factor(self) -> complex:
        """Return ``(-i)^n_response`` as a complex number."""
        return [1.0, -1j, -1.0, 1j][self.n_response % 4]

    def evaluate_coupling(
        self,
        coupling_values: dict,
        fixed_indices: dict[str, int] | None = None,
    ) -> "numpy.ndarray":
        """Substitute numeric coupling tensor values and return an array.

        Args:
            coupling_values: ``{name: array}`` mapping coupling names to
                NumPy arrays.  For a rank-3 coupling ``F``, the array
                shape should be ``(N, N, N)`` where *N* is the number
                of field components.
            fixed_indices: Optional ``{index_name: int_value}`` for
                indices pinned by diagonal constraints (e.g. observable
                component indices).

        Returns:
            NumPy array indexed by the summation indices (shape equals
            the product of component dimensions), with
            ``rational_prefactor`` already applied.
        """
        import numpy as np

        pref = self.rational_prefactor.numerator / self.rational_prefactor.denominator
        base_map: dict[str, int] = dict(fixed_indices) if fixed_indices else {}

        if not self.summation_indices:
            val = _eval_symbolic(self.coupling_sum, coupling_values, base_map)
            return np.array(pref * val)

        shape = tuple(dim for _, dim in self.summation_indices)
        result = np.zeros(shape)
        for multi_idx in np.ndindex(*shape):
            index_map = dict(base_map)
            for (name, _), val in zip(self.summation_indices, multi_idx):
                index_map[name] = val
            result[multi_idx] = _eval_symbolic(
                self.coupling_sum, coupling_values, index_map,
            )
        return pref * result

    def apply_diagonal(
        self, *, diag_R: bool = False, diag_C: bool = False,
    ) -> "DiagramTerm":
        """Return a new term with diagonal propagator constraints applied.

        Eliminates summation indices that are pinned by diagonal
        propagators and substitutes index equalities into the coupling
        expression.  The ``rational_prefactor`` is multiplied by the
        dimension of each eliminated index.

        Args:
            diag_R: Enforce diagonal response propagators.
            diag_C: Enforce diagonal correlation propagators.

        Returns:
            A new :class:`DiagramTerm` with reduced summation indices.
        """
        if not diag_R and not diag_C:
            return self

        sum_idx_set = {name for name, _ in self.summation_indices}
        sum_idx_dims = {name: dim for name, dim in self.summation_indices}

        constraints: list[tuple[str, str]] = []
        for p in self.propagators:
            if p.index_left is not None and p.index_right is not None:
                if (p.kind == "R" and diag_R) or (p.kind == "C" and diag_C):
                    if p.index_left != p.index_right:
                        constraints.append((p.index_left, p.index_right))

        if not constraints:
            return self

        # Union-find (prefer external index as root)
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
            return self

        eliminated = {idx for idx in sub if idx in sum_idx_set}

        # Update propagators
        new_props = []
        for p in self.propagators:
            il = sub.get(p.index_left, p.index_left) if p.index_left else p.index_left
            ir = sub.get(p.index_right, p.index_right) if p.index_right else p.index_right
            new_props.append(
                Propagator(p.kind, il, ir, p.spatial_left, p.spatial_right)
            )

        # Update coupling sum
        new_coupling = simplify(_apply_index_sub(self.coupling_sum, sub))

        # Update summation indices (remove eliminated)
        new_sum_indices = tuple(
            (name, dim) for name, dim in self.summation_indices
            if name not in eliminated
        )

        # Multiply prefactor by dimension for each eliminated index
        pref_num = self.rational_prefactor.numerator
        pref_den = self.rational_prefactor.denominator
        for idx_name in eliminated:
            pref_num *= sum_idx_dims[idx_name]

        return DiagramTerm(
            propagators=tuple(new_props),
            coupling_sum=new_coupling,
            rational_prefactor=Rational(pref_num, pref_den),
            integration_vars=self.integration_vars,
            summation_indices=new_sum_indices,
            n_response=self.n_response,
        )

    def to_latex(self) -> str:
        """LaTeX representation of this diagram term."""
        parts: list[str] = []
        pref = self.rational_prefactor
        if not pref.is_one:
            parts.append(pref.to_latex())
        parts.append(f"({self.coupling_sum.to_latex()})")
        for p in self.propagators:
            parts.append(p.to_latex())
        return " ".join(parts)

    def __repr__(self) -> str:
        return self.to_latex()


def compute_moment(
    observable: list[FieldOperator],
    action: Action,
    order: int,
    ito: bool = True,
    response_phase: bool = True,
    collect_topology: bool = True,
    diag_R: bool = False,
    diag_C: bool = False,
) -> PerturbativeResult:
    r"""Compute the perturbative expansion of an observable.

    Evaluates

    .. math::

       \langle \mathcal{O} \rangle_S
       = \sum_{n=0}^{N} \frac{(-1)^n}{n!}\,
         \langle \mathcal{O}\, S_{\mathrm{int}}^{\,n} \rangle_{S_0}

    up to the requested perturbative order.  In the MSR formalism the
    partition function :math:`Z = 1`, so there is no denominator.

    Args:
        observable: List of field operators defining the observable
            :math:`\mathcal{O}`.
        action: The interaction action :math:`S_{\mathrm{int}}`.
        order: Maximum perturbative order *N* to compute.
        ito: If ``True``, apply the Itô prescription
            :math:`\Theta(0)=0`: the response propagator vanishes at
            equal spatial points, :math:`R(x,x)=0`, and causal
            R-loops are eliminated.
        response_phase: If ``True``, multiply each term by
            :math:`(-\mathrm{i})^n` where *n* is the number of
            response propagators in that term, implementing the
            convention :math:`\langle\phi\,\psi\rangle =
            -\mathrm{i}\,R`.
        collect_topology: If ``True``, group terms that share the
            same propagator spatial structure and factor out the
            propagators, summing the coupling coefficients with
            appropriately permuted indices.
        diag_R: If ``True``, enforce diagonal response propagators
            :math:`R_{ij} = \delta_{ij} R`, eliminating one summation
            index per R propagator.
        diag_C: If ``True``, enforce diagonal correlation propagators
            :math:`C_{ij} = \delta_{ij} C`, eliminating one summation
            index per C propagator.

    Returns:
        A :class:`PerturbativeResult` containing order-by-order
        expressions, a combined total, and Feynman diagram information.
    """
    order_terms: dict[int, Expr] = {}
    diagrams_by_order: dict[int, list[DiagramInfo]] = {}
    dt_by_order: dict[int, list[DiagramTerm]] = {}

    for n in range(order + 1):
        sign = (-1) ** n
        fact = factorial(n)
        order_exprs: list[Expr] = []
        order_diagrams: list[DiagramInfo] = []
        order_dterms: list[DiagramTerm] = []

        if n == 0:
            # Zeroth order: just <O>_{S_0}
            wick_result, pairings = wick_contract(observable, ito=ito)
            order_exprs.append(wick_result)
            for p in pairings:
                order_diagrams.append(
                    DiagramInfo(observable, [], p, Rational(1), 0)
                )
                # Build DiagramTerm for order 0
                props = []
                for i, j in p:
                    pr = contract_pair(observable[i], observable[j], ito=ito)
                    if isinstance(pr, Propagator):
                        props.append(pr)
                if props:
                    order_dterms.append(DiagramTerm(
                        propagators=tuple(props),
                        coupling_sum=Rational(1),
                        rational_prefactor=Rational(1),
                        integration_vars=(),
                        summation_indices=(),
                        n_response=sum(1 for pr in props if pr.kind == "R"),
                    ))
        else:
            # Expand S_int^n using multinomial theorem
            for vertex_seq, multinomial_coeff in action.all_vertex_combinations(n):
                idx_ctx = IndexContext()

                # Instantiate each vertex copy
                vertex_instances = [
                    VertexInstance.instantiate(v, idx_ctx, copy_id=k)
                    for k, v in enumerate(vertex_seq)
                ]

                # Collect all operators: observable + vertex fields
                all_ops = list(observable)
                for vi in vertex_instances:
                    all_ops.extend(vi.field_operators)

                # Build prefactor: (-1)^n / n! * multinomial_coeff
                prefactor = Rational(sign * multinomial_coeff, fact)

                # Build pure coupling (without prefactor) for DiagramTerm
                coupling_syms = [vi.coupling_symbol for vi in vertex_instances]
                if len(coupling_syms) > 1:
                    pure_coupling: Expr = Product(tuple(coupling_syms))
                elif coupling_syms:
                    pure_coupling = coupling_syms[0]
                else:
                    pure_coupling = Rational(1)

                # Full coupling product (with prefactor) for the expression
                coupling_product = Product(
                    tuple([prefactor] + coupling_syms)
                )

                # Collect integration variables (vertex spatial points)
                integration_vars: frozenset[str] = frozenset(
                    var
                    for vi in vertex_instances
                    for var in vi.spatial_variables
                )

                # Build summation index info for DiagramTerm
                sum_indices: list[tuple[str, int]] = []
                for vi in vertex_instances:
                    for comp_idx in vi.component_indices:
                        for op in vi.field_operators:
                            if op.component_index == comp_idx:
                                sum_indices.append(
                                    (comp_idx, op.field.n_components)
                                )
                                break
                int_vars_sorted = tuple(sorted(integration_vars))

                if collect_topology:
                    # --- Hybrid: spatial topology + component routing ---
                    spatial_results = wick_contract_spatial(
                        all_ops, ito=ito, vertex_points=integration_vars,
                    )
                    if not spatial_results:
                        continue

                    groups: dict[
                        SpatialSignature,
                        list[tuple[list[Propagator], Pairing]],
                    ] = {}
                    pairings: list[Pairing] = []
                    for sig, (ref_props, mult, rep_pairing) in spatial_results.items():
                        routings = _enumerate_component_routings(
                            ref_props, rep_pairing, all_ops, integration_vars,
                        )
                        groups[sig] = routings
                        pairings.extend(p for _, p in routings)

                    internal_indices: set[str] = set()
                    for vi in vertex_instances:
                        internal_indices.update(vi.component_indices)

                    inner = _collect_grouped_wick(
                        groups, pure_coupling,
                        internal_indices, integration_vars,
                    )
                    if _is_zero(inner):
                        continue
                    term = Product((prefactor, inner))

                    # Extract DiagramTerm records from inner
                    for dt_props, dt_coupling in _extract_diagram_records(
                        inner
                    ):
                        order_dterms.append(DiagramTerm(
                            propagators=dt_props,
                            coupling_sum=dt_coupling,
                            rational_prefactor=prefactor,
                            integration_vars=int_vars_sorted,
                            summation_indices=tuple(sum_indices),
                            n_response=sum(
                                1 for p in dt_props if p.kind == "R"
                            ),
                        ))
                else:
                    # --- Operator-level Wick contraction ---
                    wick_result, pairings = wick_contract(all_ops, ito=ito)
                    if _is_zero(wick_result):
                        continue
                    term = Product((coupling_product, wick_result))

                # Wrap with integrals over internal spatial variables
                for vi in vertex_instances:
                    for var in vi.spatial_variables:
                        term = IntegralOver(var, term)

                # Wrap with summations over internal component indices
                for vi in vertex_instances:
                    for comp_idx in vi.component_indices:
                        field_for_idx = None
                        for op in vi.field_operators:
                            if op.component_index == comp_idx:
                                field_for_idx = op.field
                                break
                        if field_for_idx is not None:
                            term = SumOverIndex(
                                comp_idx, field_for_idx.n_components, term
                            )

                order_exprs.append(term)

                for p in pairings:
                    order_diagrams.append(
                        DiagramInfo(
                            observable,
                            vertex_instances,
                            p,
                            prefactor,
                            n,
                        )
                    )

        if order_exprs:
            raw = order_exprs[0] if len(order_exprs) == 1 else Sum(tuple(order_exprs))
            simplified = simplify(raw)
            if not collect_topology:
                # Operator-level path: group by diagram after simplification
                simplified = collect_by_diagram(simplified)
            # (Hybrid path: _collect_grouped_wick already produced coupling sums)
            if diag_R or diag_C:
                simplified = diagonal_propagators(
                    simplified, diag_R=diag_R, diag_C=diag_C,
                )
            order_terms[n] = (
                apply_response_phase(simplified) if response_phase
                else simplified
            )
        else:
            order_terms[n] = ZERO

        if diag_R or diag_C:
            order_dterms = [
                dt.apply_diagonal(diag_R=diag_R, diag_C=diag_C)
                for dt in order_dterms
            ]

        diagrams_by_order[n] = order_diagrams
        dt_by_order[n] = order_dterms

    # Total
    all_terms = [order_terms[n] for n in range(order + 1) if not _is_zero(order_terms[n])]
    if not all_terms:
        total = ZERO
    elif len(all_terms) == 1:
        total = all_terms[0]
    else:
        total = simplify(Sum(tuple(all_terms)))
        if response_phase:
            total = apply_response_phase(total)

    return PerturbativeResult(
        order_terms, total, diagrams_by_order, dt_by_order,
    )


def _collect_grouped_wick(
    groups: dict[SpatialSignature, list[tuple[list[Propagator], Pairing]]],
    pure_coupling: Expr,
    internal_indices: set[str],
    integration_vars: frozenset[str],
) -> Expr:
    """Pre-collect Wick contraction results grouped by spatial signature.

    Instead of building a huge Sum and relying on collect_by_diagram,
    this function:

    1. Computes component-index permutations within each spatial group.
    2. Merges spatial groups that share the same canonical diagram form
       (i.e. are related by integration-variable relabeling).
    3. Returns a compact Sum with one term per distinct Feynman diagram.

    The ``pure_coupling`` should be the coupling expression **without**
    the rational prefactor (so that ``_extract_diagram_records`` can
    cleanly separate coupling from prefactor).
    """
    from collections import defaultdict

    if not groups:
        return ZERO

    # --- Phase 1: Within each spatial group, collect component-index perms ---
    # Each spatial group's pairings share the same propagator spatial
    # positions but differ in component indices.
    spatial_collected: list[tuple[list[Propagator], list[dict[str, str]]]] = []

    for sig, group_entries in groups.items():
        ref_props = group_entries[0][0]
        perms: list[dict[str, str]] = [{}]  # identity for reference

        for props, _pairing in group_entries[1:]:
            comp_perm = _fast_component_match(
                ref_props, props, internal_indices,
            )
            if comp_perm is not None:
                perms.append(comp_perm)
            else:
                # Fallback: treat as its own 1-element group
                spatial_collected.append((props, [{}]))

        spatial_collected.append((ref_props, perms))

    # --- Phase 2: Group by canonical diagram form ---
    canonical_groups: dict[
        tuple[tuple[str, str, str], ...],
        list[tuple[list[Propagator], list[dict[str, str]], dict[str, str]]],
    ] = defaultdict(list)

    for ref_props, perms in spatial_collected:
        canon, mapping = _canonical_diagram_form(ref_props, integration_vars)
        canonical_groups[canon].append((ref_props, perms, mapping))

    # --- Phase 3: Merge and build expression ---
    result_terms: list[Expr] = []

    for canon, entries in canonical_groups.items():
        ref_props_0, perms_0, mapping_0 = entries[0]
        ref_inv_0 = {v: k for k, v in mapping_0.items()}

        all_coupling_terms: list[Expr] = []

        for ref_props, perms, mapping in entries:
            # Spatial relabeling: map this entry's vars to the canonical ref
            spatial_perm: dict[str, str] = {}
            for orig, canon_name in mapping.items():
                target = ref_inv_0.get(canon_name, canon_name)
                if orig != target:
                    spatial_perm[orig] = target

            # Component-index matching between this entry's ref props and
            # the canonical group's ref props
            if ref_props is ref_props_0 and not spatial_perm:
                cross_comp_perm: dict[str, str] = {}
            else:
                cross_result = _match_propagators_after_spatial(
                    ref_props_0, ref_props, spatial_perm, internal_indices,
                )
                if cross_result is None:
                    # Cannot merge — add each within-group perm separately
                    for wp in perms:
                        permuted = _apply_perm_to_coupling(pure_coupling, wp)
                        prop_expr = (
                            Product(tuple(ref_props)) if len(ref_props) > 1
                            else ref_props[0]
                        )
                        result_terms.append(Product((permuted, prop_expr)))
                    continue
                cross_comp_perm = cross_result

            cross_full = {**spatial_perm, **cross_comp_perm}

            # Compose cross-group perm with each within-group perm
            for wp in perms:
                total_perm: dict[str, str] = {}
                for k, v in wp.items():
                    total_perm[k] = cross_full.get(v, v)
                for k, v in cross_full.items():
                    if k not in total_perm:
                        total_perm[k] = v
                # Remove identity mappings
                total_perm = {k: v for k, v in total_perm.items() if k != v}
                all_coupling_terms.append(
                    _apply_perm_to_coupling(pure_coupling, total_perm)
                )

        # Build: (sum of permuted couplings) × (reference propagators)
        if len(all_coupling_terms) == 1:
            coupling_expr: Expr = all_coupling_terms[0]
        else:
            # Fast path: check if all are identical
            if all(t == all_coupling_terms[0] for t in all_coupling_terms[1:]):
                n_terms = len(all_coupling_terms)
                coupling_expr = Product(
                    (Rational(n_terms, 1), all_coupling_terms[0])
                )
            else:
                # Hash-based dedup instead of full simplify
                term_counts: dict[Expr, int] = {}
                for t in all_coupling_terms:
                    term_counts[t] = term_counts.get(t, 0) + 1
                deduped: list[Expr] = []
                for t, count in term_counts.items():
                    if count == 1:
                        deduped.append(t)
                    else:
                        deduped.append(Product((Rational(count, 1), t)))
                coupling_expr = deduped[0] if len(deduped) == 1 else Sum(tuple(deduped))

        prop_expr = (
            Product(tuple(ref_props_0)) if len(ref_props_0) > 1
            else ref_props_0[0]
        )
        result_terms.append(Product((coupling_expr, prop_expr)))

    if not result_terms:
        return ZERO
    if len(result_terms) == 1:
        return result_terms[0]
    return Sum(tuple(result_terms))


def _enumerate_component_routings(
    ref_props: list[Propagator],
    rep_pairing: Pairing,
    all_ops: list[FieldOperator],
    vertex_points: frozenset[str],
) -> list[tuple[list[Propagator], Pairing]]:
    """Enumerate all component-index routings for a given spatial topology.

    Given a spatial topology (from ``wick_contract_spatial``), permute field
    operators at each vertex point among same-type edge slots to recover all
    distinct operator-level pairings within this topology.

    Returns a list of ``(propagator_list, pairing)`` suitable for feeding
    into ``_collect_grouped_wick``.
    """
    from itertools import permutations, product as cartesian_product

    n_edges = len(rep_pairing)

    # Step 1: Build edge slot structure from the representative pairing.
    # For each edge, record which operator fills the left and right slot,
    # and classify each slot by (spatial_point, field_type).
    # slots_at_point[spatial_point][field_type] = [(edge_idx, side), ...]
    slots_at_point: dict[str, dict[str, list[tuple[int, str]]]] = {}
    ops_at_slots: dict[str, dict[str, list[int]]] = {}  # same structure but stores op indices

    for edge_idx, (op_left, op_right) in enumerate(rep_pairing):
        prop = ref_props[edge_idx]
        if prop.kind == "R":
            # Left = phi, right = psi
            phi_pt = all_ops[op_left].spatial_arg
            psi_pt = all_ops[op_right].spatial_arg
            slots_at_point.setdefault(phi_pt, {}).setdefault("phi", []).append(
                (edge_idx, "left")
            )
            ops_at_slots.setdefault(phi_pt, {}).setdefault("phi", []).append(op_left)
            slots_at_point.setdefault(psi_pt, {}).setdefault("psi", []).append(
                (edge_idx, "right")
            )
            ops_at_slots.setdefault(psi_pt, {}).setdefault("psi", []).append(op_right)
        else:
            # C edge: both sides are phi
            left_pt = all_ops[op_left].spatial_arg
            right_pt = all_ops[op_right].spatial_arg
            slots_at_point.setdefault(left_pt, {}).setdefault("phi", []).append(
                (edge_idx, "left")
            )
            ops_at_slots.setdefault(left_pt, {}).setdefault("phi", []).append(op_left)
            slots_at_point.setdefault(right_pt, {}).setdefault("phi", []).append(
                (edge_idx, "right")
            )
            ops_at_slots.setdefault(right_pt, {}).setdefault("phi", []).append(op_right)

    # Step 2: At each vertex point, enumerate permutations of operators
    # among same-type slots. Observable points are fixed.
    per_point_perms: list[list[dict[tuple[int, str], int]]] = []
    point_keys: list[tuple[str, str]] = []  # (point, field_type)

    for point in sorted(slots_at_point.keys()):
        if point not in vertex_points:
            continue  # Observable point — no permutation
        for ftype in sorted(slots_at_point[point].keys()):
            slots = slots_at_point[point][ftype]
            ops = ops_at_slots[point][ftype]
            if len(ops) <= 1:
                continue  # Only one operator — no permutation needed
            # Enumerate all permutations of ops among slots
            point_perms: list[dict[tuple[int, str], int]] = []
            for perm_ops in permutations(ops):
                mapping: dict[tuple[int, str], int] = {}
                for slot, new_op in zip(slots, perm_ops):
                    mapping[slot] = new_op
                point_perms.append(mapping)
            per_point_perms.append(point_perms)
            point_keys.append((point, ftype))

    # Step 3: Cartesian product of per-point permutations
    if not per_point_perms:
        # No permutations possible — only the reference pairing
        return [(list(ref_props), rep_pairing)]

    # Build base assignment: slot → operator (from reference pairing)
    base_assign: dict[tuple[int, str], int] = {}
    for point in slots_at_point:
        for ftype in slots_at_point[point]:
            for slot, op_idx in zip(
                slots_at_point[point][ftype], ops_at_slots[point][ftype]
            ):
                base_assign[slot] = op_idx

    seen_pairings: set[tuple[tuple[int, int], ...]] = set()
    results: list[tuple[list[Propagator], Pairing]] = []

    for combo in cartesian_product(*per_point_perms):
        # Merge all per-point slot reassignments into the base
        assign = dict(base_assign)
        for point_mapping in combo:
            assign.update(point_mapping)

        # Build new pairing and propagators from the assignment
        new_pairs: list[tuple[int, int]] = []
        new_props: list[Propagator] = []
        for edge_idx in range(n_edges):
            prop = ref_props[edge_idx]
            if prop.kind == "R":
                left_op = assign[(edge_idx, "left")]
                right_op = assign[(edge_idx, "right")]
            else:
                left_op = assign[(edge_idx, "left")]
                right_op = assign[(edge_idx, "right")]
            new_pairs.append((left_op, right_op))
            # Build propagator with the new component indices
            ol = all_ops[left_op]
            or_ = all_ops[right_op]
            new_props.append(Propagator(
                kind=prop.kind,
                index_left=ol.component_index,
                index_right=or_.component_index,
                spatial_left=ol.spatial_arg,
                spatial_right=or_.spatial_arg,
            ))

        # De-duplicate: canonicalize the pairing
        canon_pairing = tuple(sorted(
            tuple(sorted(pair)) for pair in new_pairs
        ))
        if canon_pairing in seen_pairings:
            continue
        seen_pairings.add(canon_pairing)
        results.append((new_props, tuple(new_pairs)))

    return results


def _fast_component_match(
    ref_props: list[Propagator],
    other_props: list[Propagator],
    internal_indices: set[str],
) -> dict[str, str] | None:
    """Fast component-index matching for props with identical spatial structure.

    Unlike the general ``_match_propagators_after_spatial``, this assumes
    spatial positions are already identical (no spatial perm).  It groups
    propagators by their exact ``(kind, spatial_left, spatial_right)``
    tuple and only tries permutations within tied groups.
    """
    from collections import defaultdict
    from itertools import permutations as iterperms

    # Group both lists by exact spatial key (including C directionality)
    ref_by_key: dict[tuple, list[int]] = defaultdict(list)
    other_by_key: dict[tuple, list[int]] = defaultdict(list)
    for i, p in enumerate(ref_props):
        key = (p.kind, p.spatial_left, p.spatial_right)
        ref_by_key[key].append(i)
    for i, p in enumerate(other_props):
        key = (p.kind, p.spatial_left, p.spatial_right)
        other_by_key[key].append(i)

    # Also handle C symmetry: C(x,y) matches C(y,x)
    # First try exact match; if keys don't align, try with C flipped
    if set(ref_by_key.keys()) != set(other_by_key.keys()):
        # Re-group other with C-flipped keys
        other_by_key_flip: dict[tuple, list[tuple[int, bool]]] = defaultdict(list)
        for i, p in enumerate(other_props):
            if p.kind == "C":
                # Try canonical key
                ckey = ("C", min(p.spatial_left, p.spatial_right),
                        max(p.spatial_left, p.spatial_right))
                flipped = p.spatial_left > p.spatial_right
                other_by_key_flip[ckey].append((i, flipped))
            else:
                other_by_key_flip[(p.kind, p.spatial_left, p.spatial_right)].append((i, False))

        ref_by_key_canon: dict[tuple, list[tuple[int, bool]]] = defaultdict(list)
        for i, p in enumerate(ref_props):
            if p.kind == "C":
                ckey = ("C", min(p.spatial_left, p.spatial_right),
                        max(p.spatial_left, p.spatial_right))
                flipped = p.spatial_left > p.spatial_right
                ref_by_key_canon[ckey].append((i, flipped))
            else:
                ref_by_key_canon[(p.kind, p.spatial_left, p.spatial_right)].append((i, False))

        if set(ref_by_key_canon.keys()) != set(other_by_key_flip.keys()):
            return None

        # Use canonical matching with flip tracking
        perm: dict[str, str] = {}
        for key in ref_by_key_canon:
            ri_list = ref_by_key_canon[key]
            oi_list = other_by_key_flip[key]
            if len(ri_list) != len(oi_list):
                return None
            if len(ri_list) == 1:
                ri, r_flip = ri_list[0]
                oi, o_flip = oi_list[0]
                actual_flip = r_flip != o_flip
                if not _try_add_index_perm(
                    perm, ref_props[ri], other_props[oi],
                    actual_flip, internal_indices
                ):
                    return None
            else:
                # Try all permutations of the group
                found = False
                for op in iterperms(oi_list):
                    test_perm = dict(perm)
                    ok = True
                    for (ri, r_flip), (oi, o_flip) in zip(ri_list, op):
                        actual_flip = r_flip != o_flip
                        if not _try_add_index_perm(
                            test_perm, ref_props[ri], other_props[oi],
                            actual_flip, internal_indices
                        ):
                            ok = False
                            break
                    if ok:
                        perm = test_perm
                        found = True
                        break
                if not found:
                    return None
        return perm

    # Fast path: exact key match (no C flipping needed)
    perm = {}
    for key in ref_by_key:
        ri_list = ref_by_key[key]
        oi_list = other_by_key[key]
        if len(ri_list) != len(oi_list):
            return None
        if len(ri_list) == 1:
            ri, oi = ri_list[0], oi_list[0]
            if not _try_add_index_perm(
                perm, ref_props[ri], other_props[oi], False, internal_indices
            ):
                return None
        else:
            found = False
            for op in iterperms(oi_list):
                test_perm = dict(perm)
                ok = True
                for ri, oi in zip(ri_list, op):
                    if not _try_add_index_perm(
                        test_perm, ref_props[ri], other_props[oi],
                        False, internal_indices
                    ):
                        ok = False
                        break
                if ok:
                    perm = test_perm
                    found = True
                    break
            if not found:
                return None
    return perm


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


def _apply_perm_to_coupling(coupling: Expr, perm: dict[str, str]) -> Expr:
    """Apply index permutation to a coupling product."""
    if not perm:
        return coupling
    if isinstance(coupling, Product):
        return Product(tuple(
            _apply_perm_to_coupling(f, perm) for f in coupling.factors
        ))
    if isinstance(coupling, Symbol):
        new_indices = tuple(perm.get(i, i) for i in coupling.indices)
        new_spatial = tuple(perm.get(s, s) for s in coupling.spatial_args)
        if new_indices == coupling.indices and new_spatial == coupling.spatial_args:
            return coupling
        return Symbol(coupling.name, new_indices, new_spatial)
    return coupling


def _extract_diagram_records(
    expr: Expr,
) -> list[tuple[tuple[Propagator, ...], Expr]]:
    """Extract ``(propagators, coupling_sum)`` from ``_collect_grouped_wick`` output.

    The output has a known structure: each diagram is a ``Product``
    whose factors include propagators and coupling expressions.  The
    ``Product`` constructor auto-flattens, so the factors may be
    interleaved.  We separate them by type.

    Records with identical propagator sets are merged (their couplings
    are summed).
    """
    from collections import defaultdict

    if _is_zero(expr):
        return []

    terms = list(expr.terms) if isinstance(expr, Sum) else [expr]

    raw_records: list[tuple[tuple[Propagator, ...], Expr]] = []
    for term in terms:
        if isinstance(term, Product):
            props: list[Propagator] = []
            coupling_factors: list[Expr] = []
            for f in term.factors:
                if isinstance(f, Propagator):
                    props.append(f)
                else:
                    coupling_factors.append(f)
            if not props:
                continue
            if coupling_factors:
                coupling: Expr = (
                    coupling_factors[0] if len(coupling_factors) == 1
                    else Product(tuple(coupling_factors))
                )
            else:
                coupling = Rational(1)
            raw_records.append((tuple(props), coupling))
        elif isinstance(term, Propagator):
            raw_records.append(((term,), Rational(1)))

    # Merge records with identical propagator sets
    grouped: dict[tuple[Propagator, ...], list[Expr]] = defaultdict(list)
    for props_t, coupling in raw_records:
        grouped[props_t].append(coupling)

    records: list[tuple[tuple[Propagator, ...], Expr]] = []
    for props_t, couplings in grouped.items():
        if len(couplings) == 1:
            records.append((props_t, couplings[0]))
        else:
            records.append((props_t, Sum(tuple(couplings))))
    return records


def _eval_symbolic(
    expr: Expr,
    symbol_values: dict,
    index_map: dict[str, int],
) -> float:
    """Recursively evaluate a symbolic expression with concrete values.

    Args:
        expr: The symbolic expression.
        symbol_values: ``{name: numpy_array}`` mapping coupling names
            to numeric arrays.
        index_map: ``{index_name: int_value}`` mapping component index
            names to concrete integer values.

    Returns:
        The numeric value as a float.
    """
    if isinstance(expr, Rational):
        return expr.numerator / expr.denominator
    if isinstance(expr, Symbol):
        arr = symbol_values[expr.name]
        if expr.indices:
            idx = tuple(index_map[i] for i in expr.indices)
            return float(arr[idx])
        return float(arr)
    if isinstance(expr, Product):
        result = 1.0
        for f in expr.factors:
            result *= _eval_symbolic(f, symbol_values, index_map)
        return result
    if isinstance(expr, Sum):
        return sum(
            _eval_symbolic(t, symbol_values, index_map) for t in expr.terms
        )
    raise TypeError(f"Cannot numerically evaluate {type(expr).__name__}")


def _is_zero(expr: Expr) -> bool:
    return isinstance(expr, Rational) and expr.is_zero
