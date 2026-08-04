"""Perturbative expansion driver.

Computes <O>_S = sum_{n=0}^{N} (-1)^n / n! * <O * S_int^n>_{S_0}
using Wick's theorem for each order.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import TYPE_CHECKING

import numpy as np

from .action import Action
from .expressions import (
    ZERO,
    Expr,
    ImaginaryUnit,
    IntegralOver,
    KroneckerDelta,
    Product,
    Propagator,
    Rational,
    Sum,
    SumOverIndex,
    Symbol,
    _latex_index,
    apply_response_phase,
)
from .fields import FieldOperator
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
    from .evaluate import DiagramIntegrand, SpatialStructure
    from matplotlib.figure import Figure


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

        When diagram terms are available (the default ``collect_topology=True``
        path), each order is rendered as a sum of fully-wrapped diagram
        contributions — with summation indices, integration variables,
        response phase, and prefactor all explicit.

        Falls back to the raw symbolic expression when diagram terms are
        not populated.

        Returns:
            A multi-line string with one ``O(n): <latex>`` line per
            non-zero order.
        """
        lines: list[str] = []
        for n in sorted(self.order_terms.keys()):
            dts = self.diagram_terms_by_order.get(n, [])
            if dts:
                term_strs = [dt.to_latex() for dt in dts]
                combined = " + ".join(term_strs) if len(term_strs) == 1 else \
                    " \\\\\n    + ".join(term_strs)
                lines.append(f"O({n}): {combined}")
            else:
                expr = self.order_terms[n]
                if not _is_zero(expr):
                    lines.append(f"O({n}): {expr.to_latex()}")
        return "\n".join(lines)

    def draw_diagrams(self, order: int | None = None, **kwargs) -> "Figure | None":
        """Draw Feynman diagrams using matplotlib.

        Topologically identical diagrams are drawn only once, with a
        ``×N`` multiplicity label when *N* > 1.

        Args:
            order: If given, draw only diagrams at this perturbative
                order.  Otherwise draw all diagrams.
            **kwargs: Forwarded to :class:`~sft_wick.drawing.DiagramRenderer`
                (e.g. ``figsize``).  Grid-level options accepted by
                :meth:`~sft_wick.drawing.DiagramRenderer.draw_all`
                (``ncols``, ``suptitle``, ``suptitle_kwargs``,
                ``subtitle_fn``, ``shared_legend``, ``wspace``,
                ``hspace``, ``show``, ``external_labels``) are forwarded
                there instead.

        Returns:
            The matplotlib ``Figure`` containing the rendered diagrams,
            or ``None`` when there are no diagrams.  Returning the figure
            makes the method display reliably as the final expression in
            notebooks and lets scripts call ``fig.savefig(...)``.
        """
        from collections import OrderedDict

        from .diagrams import FeynmanDiagram
        from .drawing import DiagramRenderer

        draw_all_keys = {
            "ncols",
            "suptitle",
            "suptitle_kwargs",
            "subtitle_fn",
            "shared_legend",
            "wspace",
            "hspace",
            "show",
            "external_labels",
        }
        draw_all_kwargs = {
            key: kwargs.pop(key)
            for key in list(kwargs.keys())
            if key in draw_all_keys
        }

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

        return renderer.draw_all(
            unique_diagrams,
            multiplicities=multiplicities,
            **draw_all_kwargs,
        )

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
    # Maps each non-representative internal spatial label to its
    # canonical time representative for equal_time NonLocalVertex
    # vertices. Default empty tuple means "no time aliasing"; the
    # m legs of each contributing vertex integrate independently
    # (the original sft-wick contract). When non-empty, the named
    # labels share a single time integration variable while keeping
    # independent spatial labels --- this is the equal-shell
    # cumulant case (see ``NonLocalVertex.equal_time``).
    equal_time_aliases: tuple[tuple[str, str], ...] = ()
    # ``(partner_label, leg_label)`` pairs identifying R-propagators
    # whose factor has been absorbed into an upstream ``κ^(m)_R``
    # callable (``NonLocalVertex(already_R_contracted=True)``). The
    # propagator stays in ``self.propagators`` so direction-group
    # union-find continues to identify the leg with its partner, but
    # the integrand R-product loop skips its R-factor (replaces by 1),
    # the leg time is aliased onto the partner via
    # ``equal_time_aliases``, and per-leg coupling lookups receive the
    # partner coordinates instead of the κ leg's own. Empty tuple ⇒
    # no R-absorption (original sft-wick contract).
    r_absorbed_pairs: tuple[tuple[str, str], ...] = ()
    #: Number of **external** response legs carried by the observable
    #: itself (``E_psi``).  Together with ``n_response`` this fixes the
    #: phase of the diagram: with ``n_R = sum_v n_psi(v) + E_psi`` and the
    #: package's ``(-i)^{n_R}`` factor, a diagram evaluates to
    #: ``i^{-E_psi} * (a real number)`` provided each vertex coupling
    #: carries the ``(+/-i)^{n_psi(v)}`` factor demanded by
    #: ``<phi psi> = -i R``.  So ``i^{E_psi}`` rotates the raw value onto
    #: the real axis; see :meth:`observable_phase_factor`.
    n_external_response: int = 0

    @property
    def propagator_indices(self) -> tuple[tuple[str, int], ...]:
        """Summation indices that appear in at least one propagator."""
        prop_idx_set: set[str] = set()
        for p in self.propagators:
            if p.index_left:
                prop_idx_set.add(p.index_left)
            if p.index_right:
                prop_idx_set.add(p.index_right)
        return tuple(item for item in self.summation_indices if item[0] in prop_idx_set)

    @property
    def coupling_only_indices(self) -> tuple[tuple[str, int], ...]:
        """Summation indices that appear only in the coupling sum, not in any propagator."""
        prop_idx_set: set[str] = set()
        for p in self.propagators:
            if p.index_left:
                prop_idx_set.add(p.index_left)
            if p.index_right:
                prop_idx_set.add(p.index_right)
        return tuple(item for item in self.summation_indices if item[0] not in prop_idx_set)

    def spatial_topology(self) -> list[tuple[str, str, str]]:
        """Return ``(kind, spatial_left, spatial_right)`` for each propagator."""
        return [
            (p.kind, p.spatial_left, p.spatial_right) for p in self.propagators
        ]

    def response_phase_factor(self) -> complex:
        """Return ``(-i)^n_response`` as a complex number."""
        return [1.0, -1j, -1.0, 1j][self.n_response % 4]

    def observable_phase_factor(self) -> complex:
        r"""Return ``i^{E_psi}`` — the factor that rotates this diagram's raw
        value onto the real axis.

        **Reality theorem.**  Let the diagram have vertices ``v`` with
        ``n_psi(v)`` response legs, and let the observable carry ``E_psi``
        external response legs.  Every psi must contract into an R (because
        ``<psi psi> = 0``), so ``n_R = sum_v n_psi(v) + E_psi``.  If each
        vertex coupling carries the ``(+/-i)^{n_psi(v)}`` factor demanded by
        ``<phi psi> = -i R``, then::

            (-i)^{n_R} * i^{n_R - E_psi} = i^{-E_psi}

        so the value equals ``i^{-E_psi}`` times a real number.  Multiplying by
        ``i^{E_psi}`` therefore lands exactly on the real axis.  A residual
        imaginary part means the *action* is mis-specified — an error to
        report, never a sign to guess.

        In particular ``E_psi = 0`` (an observable built only from physical
        fields) gives a strictly real value, which is why taking ``.real`` is
        correct there and taking ``abs()`` never is.
        """
        return [1.0, 1j, -1.0, -1j][self.n_external_response % 4]

    def evaluate_coupling(
        self,
        coupling_values: dict,
        fixed_indices: dict[str, int] | None = None,
    ) -> "np.ndarray":
        """Substitute numeric coupling tensor values and return an array.

        The output is indexed by **propagator indices** only.  For each
        combination of propagator index values the coupling sum is
        evaluated by summing over **coupling-only indices** (indices that
        appear in the coupling expression but in no propagator).  This
        matches the natural contraction order: fix the propagator legs,
        then sum the vertex factors internally.

        Args:
            coupling_values: ``{name: array}`` mapping coupling names to
                NumPy arrays.  For a rank-3 coupling ``F``, the array
                shape should be ``(N, N, N)`` where *N* is the number
                of field components.
            fixed_indices: Optional ``{index_name: int_value}`` for
                indices pinned by external constraints (e.g. observable
                component indices like ``{'1': 0}``).

        Returns:
            NumPy array with shape ``(dim_i0, dim_i1, ...)`` for the
            propagator indices, with ``rational_prefactor`` and the MSR
            response phase ``(-i)^n_response`` already applied.  Returns
            a 0-d array when there are no propagator indices.  The result
            is complex whenever ``n_response % 4 in {1, 3}`` or the
            coupling tensors are complex-valued.
        """
        import numpy as np

        phase = self.response_phase_factor()  # (-i)^n_response
        pref = phase * (self.rational_prefactor.numerator / self.rational_prefactor.denominator)
        base_map: dict[str, int] = dict(fixed_indices) if fixed_indices else {}
        dtype = complex if (
            isinstance(pref, complex) or
            any(np.iscomplexobj(v) for v in coupling_values.values())
        ) else float

        prop_idx = self.propagator_indices    # outer axes of result
        coup_idx = self.coupling_only_indices  # summed internally

        prop_shape = tuple(dim for _, dim in prop_idx)
        coup_shape = tuple(dim for _, dim in coup_idx)

        def _sum_coupling(prop_map: dict[str, int]) -> "np.number":
            """Evaluate coupling_sum with prop_map fixed, summing over coup_idx."""
            if not coup_idx:
                return dtype(_eval_symbolic(self.coupling_sum, coupling_values, prop_map))
            total = dtype(0)
            for cidx in np.ndindex(*coup_shape):
                index_map = {**prop_map,
                             **{name: v for (name, _), v in zip(coup_idx, cidx)}}
                total += dtype(_eval_symbolic(self.coupling_sum, coupling_values, index_map))
            return total

        if not prop_idx:
            # No propagator indices → scalar result
            val = _sum_coupling(base_map)
            return np.array(pref * val, dtype=dtype)

        result = np.zeros(prop_shape, dtype=dtype)
        for pidx in np.ndindex(*prop_shape):
            prop_map = {**base_map,
                        **{name: v for (name, _), v in zip(prop_idx, pidx)}}
            result[pidx] = _sum_coupling(prop_map)
        return pref * result

    def evaluate_coupling_batched(
        self,
        coupling_values_batched: dict,
        n_samples: int,
        fixed_indices: dict[str, int] | None = None,
    ) -> "np.ndarray":
        """Vectorised counterpart of :meth:`evaluate_coupling`.

        Same contract as :meth:`evaluate_coupling`, but every per-sample
        ``Symbol`` value carries a leading sample axis of length
        ``n_samples``.  Returns an array whose shape is
        ``(n_samples,) + prop_shape`` (or just ``(n_samples,)`` when
        there are no propagator indices).

        Args:
            coupling_values_batched: ``{name: ndarray}`` -- arrays may
                be either ``(n_samples, *kappa_shape)`` (dynamic /
                per-sample) or ``(*kappa_shape,)`` (static, broadcast
                across the sample axis).
            n_samples: Length of the sample axis.
            fixed_indices: Optional ``{index_name: int_value}`` for
                indices pinned by external constraints (e.g. observable
                component indices like ``{'a': 0}``).

        Returns:
            Complex / float array.  When this :class:`DiagramTerm` has
            propagator indices, the returned shape is
            ``(n_samples,) + prop_shape``.  Otherwise it is
            ``(n_samples,)``.

        Raises:
            NotImplementedError: If the symbolic ``coupling_sum``
                contains a node type the batched evaluator cannot
                handle.  Callers MUST catch this and fall back to a
                per-sample loop over :meth:`evaluate_coupling`.
        """
        phase = self.response_phase_factor()  # (-i)^n_response
        pref = phase * (
            self.rational_prefactor.numerator / self.rational_prefactor.denominator
        )
        base_map: dict[str, int] = dict(fixed_indices) if fixed_indices else {}

        # Decide an output dtype: complex iff phase is imaginary or any
        # supplied array is complex.
        is_complex = isinstance(pref, complex) or any(
            np.iscomplexobj(v) for v in coupling_values_batched.values()
        )
        dtype = complex if is_complex else float

        prop_idx = self.propagator_indices
        coup_idx = self.coupling_only_indices
        prop_shape = tuple(dim for _, dim in prop_idx)
        coup_shape = tuple(dim for _, dim in coup_idx)

        def _sum_coupling_batched(prop_map: dict[str, int]) -> np.ndarray:
            """Evaluate ``coupling_sum`` for a fixed ``prop_map``,
            summing over the coupling-only indices.  Returns
            ``(n_samples,)``."""
            if not coup_idx:
                val = _eval_symbolic_batched(
                    self.coupling_sum,
                    coupling_values_batched,
                    prop_map,
                    n_samples,
                )
                return np.broadcast_to(
                    np.asarray(val, dtype=dtype), (n_samples,)
                ).astype(dtype, copy=False)
            total = np.zeros(n_samples, dtype=dtype)
            for cidx in np.ndindex(*coup_shape):
                index_map = {
                    **prop_map,
                    **{name: v for (name, _), v in zip(coup_idx, cidx)},
                }
                val = _eval_symbolic_batched(
                    self.coupling_sum,
                    coupling_values_batched,
                    index_map,
                    n_samples,
                )
                total = total + np.asarray(val, dtype=dtype)
            return total

        if not prop_idx:
            val = _sum_coupling_batched(base_map)
            return (pref * val).astype(dtype, copy=False)

        result = np.zeros((n_samples,) + prop_shape, dtype=dtype)
        for pidx in np.ndindex(*prop_shape):
            prop_map = {
                **base_map,
                **{name: v for (name, _), v in zip(prop_idx, pidx)},
            }
            result[(slice(None),) + pidx] = _sum_coupling_batched(prop_map)
        return (pref * result).astype(dtype, copy=False)

    def apply_diagonal(
        self,
        *,
        diag_R: bool = False,
        diag_C: bool = False,
        iso_R: bool = False,
        iso_C: bool = False,
    ) -> "DiagramTerm":
        """Return a new term with diagonal (and optionally isotropic) propagator constraints applied.

        Eliminates summation indices that are pinned by diagonal
        propagators and substitutes index equalities into the coupling
        expression.  The delta constraint collapses the double sum into
        a single sum, so no extra factor is applied.

        With ``iso_R=True`` (implies ``diag_R=True``), the remaining equal
        component indices are stripped from R propagators entirely, expressing
        the assumption that all diagonal R entries are the same scalar R.

        Args:
            diag_R: Enforce diagonal response propagators.
            diag_C: Enforce diagonal correlation propagators.
            iso_R: Enforce isotropic (index-free) response propagators.
            iso_C: Enforce isotropic (index-free) correlation propagators.

        Returns:
            A new :class:`DiagramTerm` with reduced summation indices.
        """
        diag_R = diag_R or iso_R
        diag_C = diag_C or iso_C
        if not diag_R and not diag_C:
            return self

        sum_idx_set = {name for name, _ in self.summation_indices}

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

        # External (non-summation) index pairs that get merged by
        # the diag_R / diag_C constraint must retain a KroneckerDelta
        # factor in the coupling expression. For a SUMMATION index
        # ``i`` merged into another, the union-find rename is exact
        # because ``sum_i delta_{i, j} f(i) = f(j)``. But for an
        # OBSERVABLE index ``a`` (e.g. the labels in
        # ``phi_a(x) phi_b(y)``), ``a`` is pinned by the caller via
        # ``fixed_indices`` and is *not* summed. Dropping
        # ``delta_{a, b}`` then loses the constraint that the
        # cross-pair (a != b) order-0 contribution vanishes under
        # diag_C. Mirror the logic in ``simplify.py::diagonal_
        # propagators`` (lines ~969-977) by collecting an explicit
        # KroneckerDelta for each external-external pair.
        external_deltas: list[KroneckerDelta] = []
        seen_pairs: set[frozenset[str]] = set()
        for idx, root in sub.items():
            if idx in sum_idx_set or root in sum_idx_set:
                continue
            pair = frozenset({idx, root})
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            i1, i2 = sorted(pair)
            external_deltas.append(KroneckerDelta(i1, i2))

        # Update propagators
        new_props = []
        for p in self.propagators:
            il = sub.get(p.index_left, p.index_left) if p.index_left else p.index_left
            ir = sub.get(p.index_right, p.index_right) if p.index_right else p.index_right
            new_props.append(
                Propagator(p.kind, il, ir, p.spatial_left, p.spatial_right)
            )

        # Update coupling sum and prepend any external-external
        # deltas BEFORE the index substitution, so the deltas
        # reference the ORIGINAL (pre-merge) index names that the
        # caller's ``fixed_indices`` will pin at evaluation time.
        substituted_coupling = simplify(_apply_index_sub(self.coupling_sum, sub))
        if external_deltas:
            if isinstance(substituted_coupling, Rational) \
                    and substituted_coupling.numerator == 1 \
                    and substituted_coupling.denominator == 1:
                # Plain unity coupling -- replace it with the delta product.
                if len(external_deltas) == 1:
                    new_coupling = external_deltas[0]
                else:
                    new_coupling = Product(tuple(external_deltas))
            elif isinstance(substituted_coupling, Product):
                new_coupling = Product(
                    tuple(external_deltas) + substituted_coupling.factors
                )
            else:
                new_coupling = Product(
                    tuple(external_deltas) + (substituted_coupling,)
                )
        else:
            new_coupling = substituted_coupling

        # Update summation indices (remove eliminated)
        new_sum_indices = tuple(
            (name, dim) for name, dim in self.summation_indices
            if name not in eliminated
        )

        # Isotropic: strip equal component indices from R/C propagators
        if iso_R or iso_C:
            new_props = [
                Propagator(p.kind, None, None, p.spatial_left, p.spatial_right)
                if (
                    p.index_left is not None
                    and p.index_left == p.index_right
                    and ((p.kind == "R" and iso_R) or (p.kind == "C" and iso_C))
                )
                else p
                for p in new_props
            ]

        # Canonical rename: surviving summation indices → i_0, i_1, … in appearance order.
        # Scan propagators first (left-to-right, left-index before right-index),
        # then coupling_sum for any index that only lives there.
        surviving_set = {name for name, _ in new_sum_indices}
        ordered: list[str] = []
        ordered_set: set[str] = set()
        for p in new_props:
            for idx in (p.index_left, p.index_right):
                if idx and idx in surviving_set and idx not in ordered_set:
                    ordered.append(idx)
                    ordered_set.add(idx)
        for idx in _collect_index_order(new_coupling, surviving_set):
            if idx not in ordered_set:
                ordered.append(idx)
                ordered_set.add(idx)
        rename = {old: f"i_{j}" for j, old in enumerate(ordered)}
        if rename:
            new_coupling = _apply_index_sub(new_coupling, rename)
            new_props = [_apply_index_sub(p, rename) for p in new_props]
            dim_map = dict(new_sum_indices)
            new_sum_indices = tuple((rename[name], dim_map[name]) for name in ordered)

        return DiagramTerm(
            propagators=tuple(new_props),
            coupling_sum=new_coupling,
            rational_prefactor=self.rational_prefactor,
            integration_vars=self.integration_vars,
            summation_indices=new_sum_indices,
            n_response=self.n_response,
            equal_time_aliases=self.equal_time_aliases,
            r_absorbed_pairs=self.r_absorbed_pairs,
            n_external_response=self.n_external_response,
        )

    def to_latex(self) -> str:
        r"""Full LaTeX representation including summations, integrals, and phase.

        Renders the complete contribution::

            \sum_{i_0} ... \int dy_0 ... coeff × (coupling) × propagators

        where *coeff* combines the rational prefactor and the MSR phase
        ``(-i)^{n_R}`` into a single coefficient.
        """
        body_parts: list[str] = []

        # --- combined coefficient: rational_prefactor × (-i)^n_response ---
        coeff_latex = _coeff_latex(self.rational_prefactor, self.n_response)
        if coeff_latex:
            body_parts.append(coeff_latex)

        body_parts.append(f"({self.coupling_sum.to_latex()})")
        for p in self.propagators:
            body_parts.append(p.to_latex())
        body = " ".join(body_parts)

        # --- wrap with integrals (innermost first in the list) ---
        for var in reversed(self.integration_vars):
            body = rf"\int \mathrm{{d}}{_latex_index(var)}\, {body}"

        # --- wrap with summations (innermost first in the list) ---
        for name, dim in reversed(self.summation_indices):
            body = rf"\sum_{{{_latex_index(name)}=1}}^{{{dim}}} {body}"

        return body

    def to_latex_evaluated(
        self,
        coupling_values: dict,
        fixed_indices: dict[str, int] | None = None,
    ) -> str:
        r"""LaTeX with numerically evaluated coupling coefficients.

        Like :meth:`to_latex`, but replaces the symbolic coupling sum
        with the numerical coefficient array obtained from
        :meth:`evaluate_coupling`.  Summation indices that survive in
        the propagators are shown as explicit sums over the numerical
        coefficients; coupling-only indices are already contracted.

        Args:
            coupling_values: ``{name: numpy_array}`` for coupling tensors.
            fixed_indices: Optional pinned observable indices.

        Returns:
            LaTeX string with numerical coefficients.
        """
        coeff = self.evaluate_coupling(coupling_values, fixed_indices)
        prop_idx = self.propagator_indices

        # Format propagators
        prop_latex = " ".join(p.to_latex() for p in self.propagators)

        # Wrap with integrals
        int_wrap = ""
        for var in reversed(self.integration_vars):
            int_wrap += rf"\int \mathrm{{d}}{_latex_index(var)}\, "

        if not prop_idx:
            # Scalar coefficient
            val = complex(coeff)
            # Format nicely
            if val.imag == 0:
                val_r = val.real
                if abs(val_r - round(val_r)) < 1e-10:
                    coeff_str = str(int(round(val_r)))
                else:
                    coeff_str = f"{val_r:.4g}"
            else:
                coeff_str = f"({val:.4g})"
            if coeff_str == "1":
                coeff_str = ""
            elif coeff_str == "-1":
                coeff_str = "-"
            return f"{int_wrap}{coeff_str} {prop_latex}"

        # Array coefficient: show as sum with explicit values
        idx_names = [name for name, _ in prop_idx]
        dims = [dim for _, dim in prop_idx]

        terms: list[str] = []
        for pidx in np.ndindex(*dims):
            c = complex(coeff[pidx])
            if abs(c) < 1e-14:
                continue
            if c.imag == 0:
                c_r = c.real
                if abs(c_r - round(c_r)) < 1e-10:
                    c_str = str(int(round(c_r)))
                else:
                    c_str = f"{c_r:.4g}"
            else:
                c_str = f"({c:.4g})"

            # Substitute index values into propagators
            sub = {name: str(val + 1) for name, val in zip(idx_names, pidx)}
            prop_parts = []
            for p in self.propagators:
                il = sub.get(p.index_left, p.index_left)
                ir = sub.get(p.index_right, p.index_right)
                sl = _latex_index(p.spatial_left)
                sr = _latex_index(p.spatial_right)
                if il and ir:
                    prop_parts.append(
                        f"{p.kind}_{{{il}{ir}}}({sl}, {sr})"
                    )
                elif il or ir:
                    idx = il or ir
                    prop_parts.append(
                        f"{p.kind}_{{{idx}}}({sl}, {sr})"
                    )
                else:
                    prop_parts.append(f"{p.kind}({sl}, {sr})")

            terms.append(f"{c_str}\\," + "\\,".join(prop_parts))

        if not terms:
            return "0"

        body = " + ".join(terms)
        # Fix double signs
        body = body.replace("+ -", "- ")
        return f"{int_wrap}{body}"

    def __repr__(self) -> str:
        return self.to_latex()

    # --- Thin delegation methods for numerical evaluation pipeline ---

    def analyze_spatial(self) -> "SpatialStructure":
        """Analyze R-propagator connectivity and time orderings.

        Returns a :class:`~sft_wick.evaluate.SpatialStructure` describing
        direction identification groups, causal time orderings, and
        surviving integration variables.
        """
        from .evaluate import analyze_spatial
        return analyze_spatial(self)

    def build_integrand(
        self,
        coupling_values: dict,
        fixed_indices: dict[str, int] | None = None,
    ) -> "DiagramIntegrand":
        """Build a :class:`~sft_wick.evaluate.DiagramIntegrand` for numerical evaluation.

        Combines coupling coefficient evaluation (Step 1) with spatial
        structure analysis (Step 2) into a ready-to-evaluate object.

        Args:
            coupling_values: dict mapping coupling-symbol name → value.
                Each value is either:

                - a **numeric** ``numpy.ndarray`` of shape ``(N,)*rank``
                  (e.g. a spacetime-independent local-vertex coefficient
                  ``F``, or a placeholder for a non-local tensor whose
                  spacetime dependence is trivial),
                - a **callable** ``fn(n_list, t_list) → ndarray`` for
                  spacetime-dependent couplings (e.g. demo2's
                  ``κ^{(3)}(x₁,t₁; x₂,t₂; x₃,t₃)``).  Here ``n_list``
                  and ``t_list`` are length-``order`` sequences of the
                  vertex's ψ-leg positions and times.

                When any value is callable the integrand enters a
                **per-QMC-sample** evaluation path (see
                :class:`~sft_wick.evaluate.DynamicCouplingPromise`);
                all-ndarray values keep the fast vectorised path.
            fixed_indices: Optional pinned index values for observable
                component labels (e.g. ``{'a': 1, 'b': 1}``).

        Returns:
            A :class:`DiagramIntegrand` that can evaluate the integrand at
            specific time/direction coordinates.
        """
        from .evaluate import (
            DiagramIntegrand,
            DynamicCouplingPromise,
            analyze_spatial,
        )
        spatial = analyze_spatial(self)

        fi = dict(fixed_indices) if fixed_indices else {}

        # Detect dynamic (callable) coupling values.
        dynamic_names = {
            name: fn for name, fn in coupling_values.items()
            if callable(fn) and not isinstance(fn, np.ndarray)
        }

        # Is this diagram affected by any callable coupling?  A
        # diagram that doesn't reference the callable's symbol
        # (e.g. the order-0 ``C(x,y)`` diagram when the user has
        # also declared a K non-local vertex for higher orders) can
        # go through the fully-static fast path — the callable's
        # value is simply never needed.
        symbol_names_in_coupling = _collect_symbol_names(self.coupling_sum)
        active_dynamic = {
            name: fn for name, fn in dynamic_names.items()
            if name in symbol_names_in_coupling
        }

        if not active_dynamic:
            # No callable coupling is actually referenced — strip
            # callables from coupling_values (they'd crash the
            # static evaluator) and evaluate statically.
            static_cv = {
                name: v for name, v in coupling_values.items()
                if name not in dynamic_names
            }
            coeff = self.evaluate_coupling(static_cv, fi)
            return DiagramIntegrand(
                diagram_term=self, spatial=spatial, coupling_array=coeff,
                fixed_indices=fi,
            )

        # Dynamic path: extract each active callable symbol's
        # spatial_args from the coupling_sum so we know which ψ-leg
        # coordinates to pass at QMC time.
        spatial_args_by_name = _collect_symbol_spatial_args(
            self.coupling_sum
        )
        occurrences = _collect_symbol_occurrences(self.coupling_sum)
        for name in active_dynamic:
            if name not in spatial_args_by_name:
                raise ValueError(
                    f"coupling_values['{name}'] is callable and the "
                    f"symbol '{name}' appears in this diagram's "
                    f"coupling_sum, but carries no spatial argument, so the "
                    f"coordinates at which to evaluate it are unknown.  "
                    f"Either pass '{name}' as an ndarray (a constant "
                    f"coupling), or rebuild the Action so the vertex records "
                    f"its point."
                )
            # A callable is evaluated from a single coordinate tuple (the
            # first occurrence).  That is fine when every occurrence names the
            # same *set* of points -- the permutation-symmetrised coupling sum
            # of one non-local vertex, where the tuples differ only by leg
            # order.  (For an asymmetric kernel the leg order does matter and
            # the result is then mis-evaluated; that is a known limitation of
            # the first-occurrence rule, documented on
            # ``_collect_symbol_spatial_args``.)
            #
            # It is NOT fine when the point sets genuinely differ -- two copies
            # of the same vertex at order >= 2 sit at different times, and
            # evaluating both at the first copy's coordinates was measured
            # 4.06x wrong.  Refuse that rather than return a wrong number.
            places = occurrences.get(name, ())
            distinct_point_sets = {frozenset(pl) for pl in places}
            if len(distinct_point_sets) > 1:
                raise NotImplementedError(
                    f"coupling '{name}' is callable and occurs at "
                    f"{len(distinct_point_sets)} different sets of spacetime "
                    f"points in this diagram "
                    f"({', '.join(str(p) for p in places)}); per-occurrence "
                    f"evaluation is not implemented, and evaluating them all "
                    f"at {places[0]} would be silently wrong.  This happens "
                    f"when one vertex species appears more than once (order "
                    f">= 2).  Supported today: a callable coupling whose "
                    f"occurrences all sit at the same point set.  Pass an "
                    f"ndarray for the constant case."
                )

        static_values = {
            name: np.asarray(v) for name, v in coupling_values.items()
            if name not in dynamic_names
        }
        dynamic_names = active_dynamic  # reduce to what's active

        promise = DynamicCouplingPromise(
            diagram_term=self,
            static_values=static_values,
            dynamic_values=dict(dynamic_names),
            spatial_args_by_name=spatial_args_by_name,
            fixed_indices=fi,
        )

        # Placeholder coupling_array so the existing vectorised code
        # that reads `ig.coupling_array` doesn't error before the
        # dynamic branch is taken.  The per-sample path fills in the
        # actual values.
        prop_shape = tuple(dim for _, dim in self.propagator_indices)
        placeholder = np.zeros(prop_shape, dtype=complex)

        return DiagramIntegrand(
            diagram_term=self, spatial=spatial,
            coupling_array=placeholder,
            fixed_indices=fi,
            dynamic_coupling=promise,
        )


def _collect_r_absorbed_pairs(
    props: tuple[Propagator, ...],
    vertex_instances,
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    """Identify R-propagators that touch a leg of an
    ``already_R_contracted`` non-local vertex.

    For each such R-propagator, the leg's ψ-side gets aliased onto the
    partner's φ-side (so time and direction integration collapses), and
    the propagator is recorded as ``(partner, leg)`` in the returned
    ``r_absorbed_pairs`` tuple. The integrand evaluator skips the
    R-factor at evaluation time for any propagator in that set.

    Canonical order from :func:`~sft_wick.propagators.contract_pair`:
    ``R(spatial_left, spatial_right) = ⟨φ(spatial_left) ψ(spatial_right)⟩``,
    so the κ-leg lives at ``spatial_right`` and the partner φ at
    ``spatial_left``.

    Returns:
        ``(r_absorbed_pairs, leg_to_partner_aliases)`` — the first
        feeds ``DiagramTerm.r_absorbed_pairs``; the second is a list of
        ``(leg, partner)`` alias pairs to merge into the diagram's
        ``equal_time_aliases`` so the leg's time variable is dropped
        from integration.
    """
    absorbed_legs: set[str] = set()
    for vi in vertex_instances:
        if getattr(vi.vertex, "already_R_contracted", False):
            absorbed_legs.update(vi.spatial_variables)
    if not absorbed_legs:
        return (), ()

    pairs: list[tuple[str, str]] = []
    aliases: list[tuple[str, str]] = []
    seen_legs: set[str] = set()
    for p in props:
        if p.kind != "R":
            continue
        # canonical order: spatial_right is the ψ-side (leg).
        if p.spatial_right in absorbed_legs:
            leg = p.spatial_right
            partner = p.spatial_left
        elif p.spatial_left in absorbed_legs:
            # Defensive: if a non-canonical ordering ever slips through,
            # treat the absorbed-leg endpoint as the leg.
            leg = p.spatial_left
            partner = p.spatial_right
        else:
            continue
        if leg in seen_legs:
            # Each absorbed leg participates in exactly one R-propagator
            # (every operator pairs once in a valid Wick contraction).
            # Defensive guard against unexpected multi-pairing.
            continue
        seen_legs.add(leg)
        pairs.append((partner, leg))
        aliases.append((leg, partner))

    return tuple(sorted(pairs)), tuple(sorted(aliases))


def _collect_symbol_names(expr: Expr) -> set[str]:
    """Walk a coupling-sum tree and return the set of unique
    :class:`~sft_wick.expressions.Symbol` names present.

    Used by :meth:`DiagramTerm.build_integrand` to decide whether a
    callable coupling value is actually referenced by this
    particular diagram (so higher-order diagrams with the non-local
    vertex trigger the dynamic path, while lower-order ones keep
    the fast static path)."""
    out: set[str] = set()

    def walk(e: Expr) -> None:
        if isinstance(e, Symbol):
            out.add(e.name)
            return
        if isinstance(e, Rational):
            return
        if isinstance(e, Product):
            for f in e.factors:
                walk(f)
            return
        if isinstance(e, Sum):
            for t in e.terms:
                walk(t)
            return
        for attr in ("expr", "body", "integrand"):
            child = getattr(e, attr, None)
            if isinstance(child, Expr):
                walk(child)

    walk(expr)
    return out


def _collect_symbol_occurrences(expr: Expr) -> dict[str, tuple[tuple[str, ...], ...]]:
    """Collect **every distinct** ``spatial_args`` tuple per Symbol name.

    Companion to :func:`_collect_symbol_spatial_args`, which keeps only the
    first.  A name mapping to more than one tuple means the coupling sits at
    more than one spacetime point in this diagram — two copies of a vertex at
    order >= 2, or a permutation-symmetrised non-local coupling sum.  A
    callable coupling then cannot be evaluated from a single coordinate tuple.

    Returns ``{name: (spatial_args, ...)}`` in first-seen order; symbols with
    no spatial args are omitted.
    """
    out: dict[str, list[tuple[str, ...]]] = {}

    def walk(e: Expr) -> None:
        if isinstance(e, Symbol):
            if e.spatial_args:
                seen = out.setdefault(e.name, [])
                args = tuple(e.spatial_args)
                if args not in seen:
                    seen.append(args)
            return
        if isinstance(e, Rational):
            return
        if isinstance(e, Product):
            for f in e.factors:
                walk(f)
            return
        if isinstance(e, Sum):
            for t in e.terms:
                walk(t)
            return
        for attr in ("expr", "body", "integrand"):
            child = getattr(e, attr, None)
            if isinstance(child, Expr):
                walk(child)

    walk(expr)
    return {k: tuple(v) for k, v in out.items()}


def _collect_symbol_spatial_args(expr: Expr) -> dict[str, tuple[str, ...]]:
    """Walk a coupling-sum expression tree and collect the
    ``spatial_args`` tuple for each unique :class:`~sft_wick.expressions.Symbol`
    name.

    Used by :meth:`DiagramTerm.build_integrand` to determine, for a
    non-local vertex coupling passed as a callable, which spatial
    labels (e.g. ``y_0_0``, ``y_0_1``, ``y_0_2``) correspond to that
    vertex's ψ-legs — so that at QMC time the per-sample
    ``(n_list, t_list)`` can be reconstructed and fed into the
    callable.

    Returns ``{name: spatial_args_tuple}``.  Symbols with no
    spatial args are omitted.

    Returns the spatial_args of the symbol's **first** occurrence.

    .. warning::
       That is only valid when the name occurs at a single coordinate tuple.
       It does **not** hold in general: a permutation-symmetrised
       ``coupling_sum`` lists the same name at several leg orderings, and at
       order >= 2 two copies of one vertex live at different points.  Use
       :func:`_collect_symbol_occurrences` to detect those cases —
       :meth:`DiagramTerm.build_integrand` refuses them rather than silently
       evaluating every occurrence at the first one's coordinates.
    """
    out: dict[str, tuple[str, ...]] = {}

    def walk(e: Expr) -> None:
        if isinstance(e, Symbol):
            if e.spatial_args and e.name not in out:
                out[e.name] = tuple(e.spatial_args)
            return
        if isinstance(e, Rational):
            return
        if isinstance(e, Product):
            for f in e.factors:
                walk(f)
            return
        if isinstance(e, Sum):
            for t in e.terms:
                walk(t)
            return
        # Other Expr subclasses may wrap children on common attr names
        for attr in ("expr", "body", "integrand"):
            child = getattr(e, attr, None)
            if isinstance(child, Expr):
                walk(child)

    walk(expr)
    return out


def compute_moment(
    observable: list[FieldOperator],
    action: Action,
    order: int,
    ito: bool = True,
    response_phase: bool = True,
    collect_topology: bool = True,
    diag_R: bool = False,
    diag_C: bool = False,
    iso_R: bool = False,
    iso_C: bool = False,
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

            ``ito=False`` is a **symbolic** switch.  It keeps those
            terms in the expression tree so they can be inspected or
            rendered, but it does not change any *number*: the
            numerical layer applies :math:`\Theta(0)=0` at every R
            evaluation, and this package does not generate the
            Stratonovich functional Jacobian
            :math:`-\tfrac{1}{2}\int\mathrm{d}s\,\partial F/\partial\phi`.
            Those two omissions cancel exactly, and for additive noise
            the Itô and Stratonovich answers coincide anyway, so
            ``ito=False`` evaluates to the correct physical value — the
            same one ``ito=True`` gives.

            Do **not** "fix" this by setting
            :math:`\Theta(0)=\tfrac{1}{2}` without also emitting the
            Jacobian counter-term.  For the linear vertex that adds a
            spurious :math:`-k\,C(T,T)\,T/2`, which grows without bound
            in :math:`T` — measured at 200%/400%/800% of the exact
            answer for :math:`T=4/8/16`.  See
            ``test_F15_ito_false_changes_the_expression_not_the_number``.
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
        iso_R: If ``True`` (implies ``diag_R``), further assume all
            diagonal R entries are equal, :math:`R_{ii} = R`.  The
            component index is dropped from R propagators entirely.
        iso_C: If ``True`` (implies ``diag_C``), the same for C.

    Returns:
        A :class:`PerturbativeResult` containing order-by-order
        expressions, a combined total, and Feynman diagram information.
    """
    # E_psi: response legs carried by the observable itself.  Fixes the
    # phase needed to rotate a diagram value onto the real axis.
    _n_ext_response = sum(1 for _op in observable if _op.field.is_response)
    diag_R = diag_R or iso_R
    diag_C = diag_C or iso_C

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
                        n_external_response=_n_ext_response,
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

                # Collect equal-time alias map from any equal_time
                # NonLocalVertex instances; each maps a non-representative
                # leg label → the canonical representative whose time
                # variable is integrated. Empty when no equal_time vertex
                # is present (back-compat).
                _eq_time_alias_pairs: list[tuple[str, str]] = []
                for vi in vertex_instances:
                    for k, v in vi.equal_time_aliases or ():
                        _eq_time_alias_pairs.append((k, v))
                eq_time_aliases_tuple = tuple(sorted(_eq_time_alias_pairs))

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
                        r_absorbed_pairs, leg_aliases = (
                            _collect_r_absorbed_pairs(
                                dt_props, vertex_instances,
                            )
                        )
                        merged_aliases = (
                            tuple(sorted(
                                set(eq_time_aliases_tuple) | set(leg_aliases)
                            ))
                            if leg_aliases else eq_time_aliases_tuple
                        )
                        order_dterms.append(DiagramTerm(
                            propagators=dt_props,
                            coupling_sum=dt_coupling,
                            rational_prefactor=prefactor,
                            integration_vars=int_vars_sorted,
                            summation_indices=tuple(sum_indices),
                            n_response=sum(
                                1 for p in dt_props if p.kind == "R"
                            ),
                            equal_time_aliases=merged_aliases,
                            r_absorbed_pairs=r_absorbed_pairs,
                            n_external_response=_n_ext_response,
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
                    simplified, diag_R=diag_R, diag_C=diag_C, iso_R=iso_R, iso_C=iso_C,
                )
            order_terms[n] = (
                apply_response_phase(simplified) if response_phase
                else simplified
            )
        else:
            order_terms[n] = ZERO

        if diag_R or diag_C:
            order_dterms = [
                dt.apply_diagonal(diag_R=diag_R, diag_C=diag_C, iso_R=iso_R, iso_C=iso_C)
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


def _collect_index_order(expr: Expr, summation_set: set[str]) -> list[str]:
    """Return summation index names that appear in *expr*, in DFS first-appearance order."""
    seen: list[str] = []
    seen_set: set[str] = set()

    def _visit(e: Expr) -> None:
        if isinstance(e, Symbol):
            for idx in e.indices:
                if idx in summation_set and idx not in seen_set:
                    seen.append(idx)
                    seen_set.add(idx)
        elif isinstance(e, Propagator):
            for idx in (e.index_left, e.index_right):
                if idx and idx in summation_set and idx not in seen_set:
                    seen.append(idx)
                    seen_set.add(idx)
        elif isinstance(e, Product):
            for child in e.factors:
                _visit(child)
        elif isinstance(e, Sum):
            for child in e.terms:
                _visit(child)

    _visit(expr)
    return seen


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
        The numeric value as a float or complex.
    """
    if isinstance(expr, Rational):
        return expr.numerator / expr.denominator
    if isinstance(expr, Symbol):
        arr = symbol_values[expr.name]
        if expr.indices:
            def _resolve(i: str) -> int:
                if i in index_map:
                    return index_map[i]
                # Literal observable component (e.g. '1', '2') — 1-indexed convention
                try:
                    return int(i) - 1
                except ValueError:
                    raise KeyError(
                        f"Index '{i}' not found in index_map and is not a literal integer. "
                        "Pass it via the fixed_indices argument of evaluate_coupling()."
                    ) from None
            idx = tuple(_resolve(i) for i in expr.indices)
            val = arr[idx]
        else:
            val = arr
        val = np.asarray(val)
        if val.ndim != 0:
            if val.size == 1:
                # Tolerate a size-1 array in any shape, e.g. (1,1,1,1).
                val = val.reshape(())
            else:
                raise ValueError(
                    f"coupling '{expr.name}' carries "
                    f"{len(expr.indices)} component index/indices in this "
                    f"diagram, so indexing its value left an array of shape "
                    f"{np.shape(val)} rather than a scalar.  For a scalar "
                    f"field (n_components=1) the vertex indices are stripped "
                    f"entirely, so the coupling must be a 0-d value (e.g. "
                    f"np.array(1j)), not a rank-{np.ndim(arr)} array."
                )
        return complex(val) if np.iscomplexobj(arr) else float(val)
    if isinstance(expr, Product):
        result: complex | float = 1.0
        for f in expr.factors:
            result *= _eval_symbolic(f, symbol_values, index_map)
        return result
    if isinstance(expr, Sum):
        return sum(
            _eval_symbolic(t, symbol_values, index_map) for t in expr.terms
        )
    if isinstance(expr, KroneckerDelta):
        # Resolve each index against ``index_map``; literal observable
        # indices ('0', '1', ...) are treated as ints. The delta is
        # 1 iff the two indices resolve to the same value, else 0.
        def _resolve(i: str) -> int:
            if i in index_map:
                return int(index_map[i])
            try:
                return int(i)
            except ValueError:
                raise KeyError(
                    f"KroneckerDelta index {i!r} not found in index_map "
                    f"and is not a literal integer. Pass it via the "
                    f"fixed_indices argument of evaluate_coupling()."
                ) from None
        return 1.0 if _resolve(expr.index1) == _resolve(expr.index2) else 0.0
    if isinstance(expr, ImaginaryUnit):
        return 1j
    raise TypeError(f"Cannot numerically evaluate {type(expr).__name__}")


def _eval_symbolic_batched(
    expr: Expr,
    symbol_values_batched: dict,
    index_map: dict[str, int],
    n_samples: int,
) -> "np.ndarray":
    """Vectorised counterpart of :func:`_eval_symbolic`.

    Walks the symbolic expression tree once and returns an array of
    shape ``(n_samples,)`` instead of a scalar.  ``Symbol`` values in
    ``symbol_values_batched`` may carry a leading sample axis (so the
    array shape is ``(n_samples,) + (N,) * rank``); static (non-batched)
    arrays of shape ``(N,) * rank`` are also accepted and broadcast
    automatically.  All other node types collapse to scalars and are
    promoted to ``(n_samples,)`` only when the parent ``Product`` /
    ``Sum`` mixes them with a batched child.

    For node types that cannot be cheaply vectorised
    (``Propagator``, ``IntegralOver``, ``SumOverIndex``, ``DiracDelta``)
    a :class:`NotImplementedError` is raised so the caller can fall back
    to the scalar per-sample loop.  These nodes never appear in the
    ``coupling_sum`` of a :class:`DiagramTerm` (couplings are strictly
    products / sums of ``Symbol``, ``KroneckerDelta``, ``Rational`` and
    ``ImaginaryUnit`` after :func:`apply_response_phase`), so the
    fallback only triggers for unusual user inputs.

    Args:
        expr: The symbolic expression (typically a ``DiagramTerm``'s
            ``coupling_sum``).
        symbol_values_batched: ``{name: ndarray}`` mapping coupling
            symbol names to either ``(n_samples, *kappa_shape)`` arrays
            (dynamic / per-sample) or ``(*kappa_shape,)`` arrays
            (static, broadcast across the sample axis).
        index_map: ``{index_name: int_value}`` for component indices
            pinned by the caller (e.g. propagator-index outer loop or
            ``fixed_indices``).
        n_samples: Length of the sample axis.

    Returns:
        A complex or float ``(n_samples,)`` array.
    """
    if isinstance(expr, Rational):
        return expr.numerator / expr.denominator  # scalar; broadcast later
    if isinstance(expr, ImaginaryUnit):
        return 1j  # scalar; broadcast later
    if isinstance(expr, Symbol):
        arr = symbol_values_batched[expr.name]
        if not expr.indices:
            # Scalar symbol -- just return as-is.
            return arr
        # Resolve component indices to integers (literal '1' → 0 etc.)
        def _resolve(i: str) -> int:
            if i in index_map:
                return index_map[i]
            try:
                return int(i) - 1
            except ValueError:
                raise KeyError(
                    f"Index '{i}' not found in index_map and is not a "
                    f"literal integer. Pass it via the fixed_indices "
                    f"argument of evaluate_coupling()."
                ) from None
        idx = tuple(_resolve(i) for i in expr.indices)
        # Detect whether ``arr`` carries a leading sample axis.
        # rank of the static tensor is len(expr.indices); compare
        # ``arr.ndim`` to decide whether to prepend ``slice(None)``.
        if arr.ndim == len(idx) + 1:
            # Batched: arr shape is (n_samples,) + (N,)*rank.
            return arr[(slice(None),) + idx]
        if arr.ndim == len(idx):
            # Static: shape is (N,)*rank — return scalar; caller's
            # Product / Sum logic will broadcast.
            return arr[idx]
        raise ValueError(
            f"Symbol {expr.name!r} has rank {len(idx)} but the supplied "
            f"array has shape {arr.shape}; expected either rank or "
            f"rank+1 (with leading sample axis of length {n_samples})."
        )
    if isinstance(expr, KroneckerDelta):
        def _resolve(i: str) -> int:
            if i in index_map:
                return int(index_map[i])
            try:
                return int(i)
            except ValueError:
                raise KeyError(
                    f"KroneckerDelta index {i!r} not found in index_map "
                    f"and is not a literal integer. Pass it via the "
                    f"fixed_indices argument of evaluate_coupling()."
                ) from None
        return 1.0 if _resolve(expr.index1) == _resolve(expr.index2) else 0.0
    if isinstance(expr, Product):
        result: object = 1.0
        for f in expr.factors:
            result = result * _eval_symbolic_batched(
                f, symbol_values_batched, index_map, n_samples,
            )
        return result
    if isinstance(expr, Sum):
        terms = [
            _eval_symbolic_batched(
                t, symbol_values_batched, index_map, n_samples,
            )
            for t in expr.terms
        ]
        if not terms:
            return 0.0
        total = terms[0]
        for t in terms[1:]:
            total = total + t
        return total
    raise NotImplementedError(
        f"_eval_symbolic_batched: node type {type(expr).__name__} "
        "is not vectorisable; caller should fall back to the scalar "
        "per-sample loop."
    )


def _is_zero(expr: Expr) -> bool:
    return isinstance(expr, Rational) and expr.is_zero


def _coeff_latex(pref: Rational, n_response: int) -> str:
    r"""Render ``rational_prefactor × (-i)^{n_response}`` as a single LaTeX token.

    Combines the sign of the rational with the phase so that the output
    is a clean fraction like ``\frac{\mathrm{i}}{6}`` rather than two
    separate tokens ``\mathrm{i} -\frac{1}{6}``.
    """
    from fractions import Fraction

    # Phase by n_response mod 4:
    #   0 → +1,  1 → -i,  2 → -1,  3 → +i
    # Represent as (real_sign, is_imag):
    #   0 → (+1, False), 1 → (-1, True), 2 → (-1, False), 3 → (+1, True)
    phase_sign = [1, -1, -1, 1][n_response % 4]
    is_imag = (n_response % 2) == 1

    # Effective rational = pref × phase_sign  (absorb sign into numerator)
    f = Fraction(pref.numerator * phase_sign, pref.denominator)
    num, den = abs(f.numerator), f.denominator
    negative = f < 0

    # Build the numerator string
    if is_imag and num == 1:
        num_str = r"\mathrm{i}"
    elif is_imag:
        num_str = rf"{num}\,\mathrm{{i}}"
    else:
        num_str = str(num)

    # Assemble
    sign = "-" if negative else ""
    if num == 0:
        return "0"
    if den == 1 and num_str == "1" and not is_imag:
        # coefficient is ±1 → omit (sign is still prepended if negative)
        return f"{sign}1" if negative else ""
    if den == 1:
        return f"{sign}{num_str}"
    return rf"{sign}\frac{{{num_str}}}{{{den}}}"


# =====================================================================
# Fast numerical path: nauty + einsum
# =====================================================================


def compute_moment_numerical(
    observable: list[FieldOperator],
    action: Action,
    order: int,
    coupling_values: dict,
    fixed_indices: dict[str, int] | None = None,
    ito: bool = True,
    diag_R: bool = False,
    diag_C: bool = False,
    iso_R: bool = False,
    iso_C: bool = False,
    n_jobs: int = 1,
) -> dict[int, list[DiagramTerm]]:
    r"""Compute diagram terms using nauty graph isomorphism.

    A faster alternative to :func:`compute_moment` that replaces the
    :math:`k!` brute-force canonical form search with the nauty graph
    isomorphism algorithm (via ``pynauty``), enabling order-6
    calculations that were previously infeasible.

    Optionally parallelizes the nauty canonicalization and component
    routing steps using ``joblib`` (install with ``pip install
    sft-wick[parallel]``).

    Args:
        observable: Field operators defining the observable.
        action: The interaction action.
        order: Maximum perturbative order.
        coupling_values: ``{name: numpy_array}`` mapping coupling names
            to numeric tensors.
        fixed_indices: ``{index_label: int_value}`` for observable
            component indices (e.g. ``{'a': 1, 'b': 1}``).
        ito: Apply Itô prescription (default True).
        diag_R: Enforce diagonal R propagators.
        diag_C: Enforce diagonal C propagators.
        iso_R: Isotropic R (implies ``diag_R``).
        iso_C: Isotropic C (implies ``diag_C``).
        n_jobs: Number of parallel workers for nauty canonicalization
            and component routing.  ``1`` = sequential (default),
            ``-1`` = use all CPU cores.  Requires ``joblib``.

    Returns:
        ``{order_n: [DiagramTerm, ...]}`` for each order with
        non-zero contributions.  Use :meth:`DiagramTerm.build_integrand`
        for numerical evaluation.
    """
    # E_psi: response legs carried by the observable itself.  Fixes the
    # phase needed to rotate a diagram value onto the real axis.
    _n_ext_response = sum(1 for _op in observable if _op.field.is_response)
    from collections import defaultdict

    from .simplify import _canonical_key_nauty

    diag_R = diag_R or iso_R
    diag_C = diag_C or iso_C
    fixed_indices = fixed_indices or {}

    result: dict[int, list[DiagramTerm]] = {}

    for n in range(order + 1):
        sign = (-1) ** n
        fact = factorial(n)
        integrands: list[DiagramIntegrand] = []

        if n == 0:
            # Zeroth order: <O>_{S_0}
            if len(observable) >= 2:
                _, pairings = wick_contract(observable, ito=ito)
                for p in pairings:
                    props = []
                    for i, j in p:
                        pr = contract_pair(observable[i], observable[j], ito=ito)
                        if isinstance(pr, Propagator):
                            props.append(pr)
                    if props:
                        dt = DiagramTerm(
                            propagators=tuple(props),
                            coupling_sum=Rational(1),
                            rational_prefactor=Rational(1),
                            integration_vars=(),
                            summation_indices=(),
                            n_response=sum(1 for pr in props if pr.kind == "R"),
                            n_external_response=_n_ext_response,
                        )
                        if diag_R or diag_C:
                            dt = dt.apply_diagonal(
                                diag_R=diag_R, diag_C=diag_C,
                                iso_R=iso_R, iso_C=iso_C,
                            )
                        integrands.append(dt)
        else:
            for vertex_seq, mc in action.all_vertex_combinations(n):
                idx_ctx = IndexContext()
                vis = [
                    VertexInstance.instantiate(v, idx_ctx, copy_id=k)
                    for k, v in enumerate(vertex_seq)
                ]

                all_ops = list(observable)
                for vi in vis:
                    all_ops.extend(vi.field_operators)

                integration_vars = frozenset(
                    var for vi in vis for var in vi.spatial_variables
                )

                # Collect equal-time alias map (see collect_topology
                # branch above for the full motivation).
                _eq_time_alias_pairs2: list[tuple[str, str]] = []
                for vi in vis:
                    for k, v in vi.equal_time_aliases or ():
                        _eq_time_alias_pairs2.append((k, v))
                eq_time_aliases_tuple2 = tuple(
                    sorted(_eq_time_alias_pairs2)
                )

                # Spatial topology enumeration
                spatial_results = wick_contract_spatial(
                    all_ops, ito=ito, vertex_points=integration_vars,
                )
                if not spatial_results:
                    continue

                # Group topologies by nauty canonical form, then build
                # symbolic DiagramTerms (compatible with build_integrand).
                # Combined pipeline: for each topology, compute nauty
                # canonical key + pruned component routing in one pass.
                # Results are grouped by canonical key afterward.

                # Build summation index info (once)
                sum_indices: list[tuple[str, int]] = []
                for vi in vis:
                    for comp_idx in vi.component_indices:
                        for op in vi.field_operators:
                            if op.component_index == comp_idx:
                                sum_indices.append(
                                    (comp_idx, op.field.n_components)
                                )
                                break

                prefactor = Rational(sign * mc, fact)
                int_vars_sorted = tuple(sorted(integration_vars))
                topo_list = list(spatial_results.values())

                def _process_topology(ref_props, mult, rep_pairing):
                    """Nauty key + pruned routing for one topology."""
                    key = _canonical_key_nauty(
                        ref_props, integration_vars
                    )
                    routings = [
                        props for props, _pairing in
                        _enumerate_component_routings(
                            ref_props, rep_pairing, all_ops,
                            integration_vars,
                        )
                    ]
                    return key, routings

                # Run pipeline (parallel if beneficial)
                if n_jobs != 1 and len(topo_list) > 100:
                    try:
                        from joblib import Parallel, delayed
                        topo_results = Parallel(n_jobs=n_jobs)(
                            delayed(_process_topology)(rp, m, p)
                            for rp, m, p in topo_list
                        )
                    except ImportError:
                        topo_results = [
                            _process_topology(rp, m, p)
                            for rp, m, p in topo_list
                        ]
                else:
                    topo_results = [
                        _process_topology(rp, m, p)
                        for rp, m, p in topo_list
                    ]

                # Group by canonical key
                canon_groups: dict[bytes, list] = defaultdict(list)
                ref_props_by_key: dict[bytes, list] = {}
                for (rp, _m, _p), (key, routings) in zip(
                    topo_list, topo_results
                ):
                    if routings:
                        canon_groups[key].extend(routings)
                        if key not in ref_props_by_key:
                            ref_props_by_key[key] = rp

                # Build one DiagramTerm per routing.
                # Each routing has its own propagator component indices
                # and the coupling uses the vertex's static field ordering
                # (the coupling IS a vertex property; the routing only
                # changes which propagator carries which component index).
                for key, all_routing_props in canon_groups.items():
                    # Build the (static) coupling symbol product
                    coupling_syms: list[Expr] = []
                    for vi in vis:
                        idx_vals = tuple(
                            op.component_index
                            for op in vi.field_operators
                        )
                        coupling_syms.append(Symbol(
                            name=vi.vertex.coupling,
                            indices=idx_vals,
                            spatial_args=(),
                        ))
                    coupling_expr: Expr = (
                        Product(tuple(coupling_syms))
                        if len(coupling_syms) > 1
                        else coupling_syms[0]
                    )

                    for props in all_routing_props:
                        props_tuple = tuple(props)
                        r_absorbed_pairs2, leg_aliases2 = (
                            _collect_r_absorbed_pairs(props_tuple, vis)
                        )
                        merged_aliases2 = (
                            tuple(sorted(
                                set(eq_time_aliases_tuple2) | set(leg_aliases2)
                            ))
                            if leg_aliases2 else eq_time_aliases_tuple2
                        )
                        integrands.append(DiagramTerm(
                            propagators=props_tuple,
                            coupling_sum=coupling_expr,
                            rational_prefactor=prefactor,
                            integration_vars=int_vars_sorted,
                            summation_indices=tuple(sum_indices),
                            n_response=sum(
                                1 for p in props_tuple
                                if p.kind == "R"
                            ),
                            equal_time_aliases=merged_aliases2,
                            r_absorbed_pairs=r_absorbed_pairs2,
                            n_external_response=_n_ext_response,
                        ))

        # Apply diagonal constraints
        if (diag_R or diag_C) and integrands:
            integrands = [
                dt.apply_diagonal(
                    diag_R=diag_R, diag_C=diag_C,
                    iso_R=iso_R, iso_C=iso_C,
                )
                for dt in integrands
            ]

        if integrands:
            result[n] = integrands

    return result
