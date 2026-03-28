"""Numerical evaluation pipeline for DiagramTerm objects.

Provides the 4-step workflow:
1. Coupling coefficients — handled by DiagramTerm.evaluate_coupling() (existing)
2. Spatial structure analysis — analyze_spatial()
3. Propagator evaluation — PropagatorModel + PropagatorCache
4. Contraction & integration — DiagramIntegrand
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import prod
from typing import Any, Callable

import numpy as np

from .expressions import Propagator

# ---------------------------------------------------------------------------
# Step 2: Spatial Structure Analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpatialStructure:
    """Result of analyzing R-propagator connectivity and time orderings.

    R propagators of the form ``δ(n−n') Θ(t−t') R_time(t,t')`` identify
    directions along R-chains and impose causal time orderings.
    """

    #: Groups of spatial points connected by R propagators (share direction).
    direction_groups: tuple[frozenset[str], ...]

    #: spatial_point → representative direction variable name (e.g. ``'n_x'``).
    direction_map: dict[str, str]

    #: Time ordering pairs: ``(earlier_point, later_point)`` from R causality.
    #: ``R(a, b)`` with ``Θ(t_a − t_b)`` means ``t_b ≤ t_a``.
    time_orderings: tuple[tuple[str, str], ...]

    #: R propagators as ``(spatial_left, spatial_right)`` pairs.
    r_propagators: tuple[tuple[str, str], ...]

    #: C propagators: ``(spatial_left, spatial_right, index_left, index_right)``.
    c_propagators: tuple[tuple[str, str, str | None, str | None], ...]

    #: Time integration variables, topologically sorted (innermost first).
    time_integration_vars: tuple[str, ...]

    #: Surviving direction integration variables (one per R-component that
    #: contains only integration points — typically empty).
    direction_integration_vars: tuple[str, ...]

    #: External (non-integration) spatial points.
    external_points: tuple[str, ...]


def _topological_sort_times(
    integration_vars: tuple[str, ...],
    time_orderings: list[tuple[str, str]],
) -> list[str]:
    """Topological sort of integration time variables using causal ordering.

    Returns variables ordered so that the *innermost* integration variable
    (the one with the tightest bounds) comes first.  This is the reverse
    of the DAG order: leaves (earliest times) first.
    """
    # Build adjacency: earlier → later
    adj: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = {v: 0 for v in integration_vars}
    int_set = set(integration_vars)

    for earlier, later in time_orderings:
        if earlier in int_set and later in int_set:
            adj[later].append(earlier)
            in_degree[earlier] = in_degree.get(earlier, 0) + 1

    # Kahn's algorithm — start from nodes with in_degree 0 (latest times)
    queue = [v for v in integration_vars if in_degree[v] == 0]
    result: list[str] = []
    while queue:
        queue.sort()  # deterministic tie-breaking
        node = queue.pop(0)
        result.append(node)
        for neighbor in adj.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # result is latest-first; reverse to get innermost (earliest) first
    result.reverse()
    return result


def analyze_spatial(dt: "DiagramTerm") -> SpatialStructure:
    """Analyze a DiagramTerm's propagator topology.

    Determines direction identification groups from R-propagator connectivity,
    time ordering constraints from R causality, and which integration variables
    survive after δ-function elimination.
    """
    integration_set = set(dt.integration_vars)

    # Classify propagators
    r_props: list[tuple[str, str]] = []
    c_props: list[tuple[str, str, str | None, str | None]] = []
    for p in dt.propagators:
        if p.kind == "R":
            r_props.append((p.spatial_left, p.spatial_right))
        else:
            c_props.append((p.spatial_left, p.spatial_right, p.index_left, p.index_right))

    # --- Direction groups via union-find on R propagators ---
    all_points: set[str] = set()
    for p in dt.propagators:
        all_points.add(p.spatial_left)
        all_points.add(p.spatial_right)

    parent: dict[str, str] = {p: p for p in all_points}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Prefer external point as root
            if rb not in integration_set and ra in integration_set:
                parent[ra] = rb
            else:
                parent[rb] = ra

    for sl, sr in r_props:
        union(sl, sr)

    # Build groups
    groups_dict: dict[str, set[str]] = defaultdict(set)
    for p in all_points:
        groups_dict[find(p)].add(p)

    direction_groups = tuple(frozenset(g) for g in groups_dict.values())

    # Direction map: each point → n_{representative}
    direction_map: dict[str, str] = {}
    direction_integration_vars: list[str] = []
    for group in direction_groups:
        # Pick representative: prefer external point, then lexicographic
        external = sorted(p for p in group if p not in integration_set)
        if external:
            rep = external[0]
        else:
            rep = sorted(group)[0]
            direction_integration_vars.append(rep)
        dir_name = f"n_{rep}"
        for p in group:
            direction_map[p] = dir_name

    # --- Time orderings from R causality ---
    # R(a, b): Θ(t_a − t_b) → t_b ≤ t_a → (earlier=b, later=a)
    time_orderings: list[tuple[str, str]] = []
    for sl, sr in r_props:
        time_orderings.append((sr, sl))  # (earlier, later)

    # --- Topological sort of integration time variables ---
    sorted_time_vars = _topological_sort_times(
        dt.integration_vars, time_orderings
    )

    # --- External points ---
    external_points = tuple(sorted(p for p in all_points if p not in integration_set))

    return SpatialStructure(
        direction_groups=direction_groups,
        direction_map=direction_map,
        time_orderings=tuple(time_orderings),
        r_propagators=tuple(r_props),
        c_propagators=tuple(c_props),
        time_integration_vars=tuple(sorted_time_vars),
        direction_integration_vars=tuple(sorted(direction_integration_vars)),
        external_points=external_points,
    )


# ---------------------------------------------------------------------------
# Step 3: Propagator Model and Cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PropagatorModel:
    """Physical propagator functions for numerical evaluation.

    The R propagator has the form::

        R_{ab}(n,t; n',t') = δ(n−n') Θ(t−t') R_time(t,t') δ_{ab}   [iso_R]
        R_{ab}(n,t; n',t') = δ(n−n') Θ(t−t') R_time_ab(t,t')        [general]

    The C propagator is computed from R and the 2-point cumulant κ^(2)::

        C_{ab}(n₁,t₁; n₂,t₂) = ∫∫ R_time(t₁,λ') κ_{ab}(n₁,λ'; n₂,λ'')
                                      R_time(t₂,λ'') dλ' dλ''

    Attributes:
        R_time: Callable ``(t_left, t_right) → scalar`` when ``iso_R=True``,
            or ``(t_left, t_right) → (N, N) array`` otherwise.
        kappa2: Callable ``(n1, t1, n2, t2) → (N, N) array``.
            The 2-point cumulant.
        n_components: Number of field components.
        iso_R: Whether R is isotropic (proportional to identity).
        diag_C: Whether C is diagonal in component indices.
        t_min: Lower bound for the λ integrals in C computation.
    """

    R_time: Callable
    kappa2: Callable
    n_components: int
    iso_R: bool = True
    diag_C: bool = False
    t_min: float = 0.0


class PropagatorCache:
    """Evaluates and caches propagator values.

    Computes C via double integration of ``R · κ · R`` and caches results
    to avoid redundant quadrature.

    Args:
        model: The physical propagator model.
        quad_opts: Options passed to ``scipy.integrate.dblquad``
            (e.g. ``{'epsabs': 1e-8, 'epsrel': 1e-8}``).
    """

    def __init__(
        self,
        model: PropagatorModel,
        quad_opts: dict | None = None,
    ):
        self.model = model
        self.quad_opts = quad_opts or {}
        self._c_cache: dict[tuple, np.ndarray] = {}

    def R_time(self, t_left: float, t_right: float) -> float | np.ndarray:
        """Evaluate R_time(t_left, t_right).

        Returns scalar if ``model.iso_R=True``, else ``(N, N)`` array.
        The Θ function is NOT enforced here — the integration domain
        handles causality.
        """
        return self.model.R_time(t_left, t_right)

    def R_product(
        self,
        r_pairs: tuple[tuple[str, str], ...],
        times: dict[str, float],
    ) -> float:
        """Product of R_time over all R propagator pairs.

        Only valid when ``model.iso_R=True`` (R is scalar).
        """
        result = 1.0
        for sl, sr in r_pairs:
            result *= float(self.R_time(times[sl], times[sr]))
        return result

    def C_value(
        self,
        n1: Any,
        t1: float,
        n2: Any,
        t2: float,
    ) -> np.ndarray:
        """Compute the full C matrix ``C_{ab}(n1, t1; n2, t2)``.

        Uses ``scipy.integrate.dblquad``::

            C_{ab} = ∫_{t_min}^{t1} dλ' ∫_{t_min}^{t2} dλ''
                     R_time(t1,λ') κ_{ab}(n1,λ'; n2,λ'') R_time(t2,λ'')

        Returns:
            ``(N, N)`` array.
        """
        from scipy.integrate import dblquad

        m = self.model
        N = m.n_components
        t_min = m.t_min

        # Check cache
        cache_key = (id(n1) if isinstance(n1, np.ndarray) else n1,
                     t1, id(n2) if isinstance(n2, np.ndarray) else n2, t2)
        if cache_key in self._c_cache:
            return self._c_cache[cache_key]

        C_mat = np.zeros((N, N))

        if m.diag_C:
            # Only compute diagonal
            for a in range(N):
                def integrand(lam2: float, lam1: float, _a: int = a) -> float:
                    r1 = float(m.R_time(t1, lam1)) if m.iso_R else float(m.R_time(t1, lam1)[_a, _a])
                    kappa_mat = m.kappa2(n1, lam1, n2, lam2)
                    r2 = float(m.R_time(t2, lam2)) if m.iso_R else float(m.R_time(t2, lam2)[_a, _a])
                    return r1 * float(kappa_mat[_a, _a]) * r2

                val, _ = dblquad(
                    integrand,
                    t_min, t1,   # outer: lam1 bounds
                    t_min, t2,   # inner: lam2 bounds
                    **self.quad_opts,
                )
                C_mat[a, a] = val
        else:
            for a in range(N):
                for b in range(N):
                    def integrand(lam2: float, lam1: float, _a: int = a, _b: int = b) -> float:
                        if m.iso_R:
                            r1 = float(m.R_time(t1, lam1))
                            r2 = float(m.R_time(t2, lam2))
                        else:
                            r1 = float(m.R_time(t1, lam1)[_a, _a])
                            r2 = float(m.R_time(t2, lam2)[_b, _b])
                        kappa_mat = m.kappa2(n1, lam1, n2, lam2)
                        return r1 * float(kappa_mat[_a, _b]) * r2

                    val, _ = dblquad(
                        integrand,
                        t_min, t1,
                        t_min, t2,
                        **self.quad_opts,
                    )
                    C_mat[a, b] = val

        self._c_cache[cache_key] = C_mat
        return C_mat

    def C_diagonal(
        self,
        n: Any,
        t1: float,
        n_prime: Any | None = None,
        t2: float | None = None,
    ) -> np.ndarray:
        """Return diagonal ``[C_{00}, C_{11}, …]`` as a 1D array.

        If ``n_prime`` and ``t2`` are None, uses equal-point ``C(n,t; n,t)``.
        """
        if n_prime is None:
            n_prime = n
        if t2 is None:
            t2 = t1
        C_mat = self.C_value(n, t1, n_prime, t2)
        return np.diag(C_mat)

    def clear_cache(self) -> None:
        """Clear the C value cache."""
        self._c_cache.clear()


# ---------------------------------------------------------------------------
# Step 4: Contraction & Integration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiagramIntegrand:
    """A diagram's contribution decomposed for numerical integration.

    Combines coupling coefficients (Step 1) with spatial analysis (Step 2)
    for efficient numerical evaluation.

    The integrand has the factored form (when ``iso_R=True``)::

        R_product(times) × Σ_{prop_idx} coeff[idx] × Π_C C_values[idx]
    """

    #: The underlying DiagramTerm.
    diagram_term: "DiagramTerm"

    #: Spatial analysis result.
    spatial: SpatialStructure

    #: Coupling coefficient array from ``evaluate_coupling()``.
    coupling_array: np.ndarray

    def evaluate(
        self,
        times: dict[str, float],
        directions: dict[str, Any],
        cache: PropagatorCache,
    ) -> complex:
        """Evaluate the integrand at specific time + direction coordinates.

        Args:
            times: ``{spatial_point: time_value}`` for ALL spatial points.
            directions: ``{direction_var: value}`` for independent direction
                variables (one per R-connected component, keyed by
                the representative name from ``spatial.direction_map``).
            cache: A :class:`PropagatorCache` for propagator evaluation.

        Returns:
            Scalar value of the integrand (complex if coupling is complex).
        """
        dt = self.diagram_term
        spatial = self.spatial
        coeff = self.coupling_array
        model = cache.model

        # --- R product (scalar when iso_R) ---
        r_val = cache.R_product(spatial.r_propagators, times)

        # --- Evaluate C propagators ---
        prop_idx = dt.propagator_indices
        if not prop_idx:
            # No propagator indices → scalar coupling, evaluate C without indices
            c_val = 1.0
            for sp_l, sp_r, _, _ in spatial.c_propagators:
                dir_l = spatial.direction_map[sp_l]
                dir_r = spatial.direction_map[sp_r]
                n_l = directions.get(dir_l, directions.get(sp_l))
                n_r = directions.get(dir_r, directions.get(sp_r))
                C_mat = cache.C_value(n_l, times[sp_l], n_r, times[sp_r])
                c_val *= C_mat.trace()  # scalar contraction for isotropic C
            return complex(r_val * float(coeff) * c_val)

        # --- Map each C propagator to its propagator-index axis ---
        idx_names = [name for name, _ in prop_idx]
        idx_name_to_axis = {name: ax for ax, name in enumerate(idx_names)}

        if model.iso_R and model.diag_C:
            return self._evaluate_diag_fast(
                r_val, times, directions, cache, idx_names, idx_name_to_axis
            )
        else:
            return self._evaluate_general(
                r_val, times, directions, cache, idx_names, idx_name_to_axis
            )

    def _evaluate_diag_fast(
        self,
        r_val: float,
        times: dict[str, float],
        directions: dict[str, Any],
        cache: PropagatorCache,
        idx_names: list[str],
        idx_name_to_axis: dict[str, int],
    ) -> complex:
        """Fast path for iso_R + diag_C case.

        Each C propagator contributes a diagonal vector; contraction is
        element-wise multiplication then summation.
        """
        spatial = self.spatial
        coeff = self.coupling_array
        n_axes = coeff.ndim

        contracted = coeff.copy()
        for sp_l, sp_r, il, ir in spatial.c_propagators:
            dir_l = spatial.direction_map[sp_l]
            dir_r = spatial.direction_map[sp_r]
            n_l = directions.get(dir_l, directions.get(sp_l))
            n_r = directions.get(dir_r, directions.get(sp_r))
            c_diag = cache.C_diagonal(n_l, times[sp_l], n_r, times[sp_r])

            # Diagonal: il == ir (after apply_diagonal).  Find the axis.
            idx_name = il  # = ir for diagonal propagators
            if idx_name is None:
                # Isotropic C (iso_C): no index → scalar trace
                contracted = contracted * c_diag.sum()
                continue
            axis = idx_name_to_axis.get(idx_name)
            if axis is None:
                # Index not in propagator_indices (shouldn't happen normally)
                contracted = contracted * c_diag.sum()
                continue

            # Broadcast c_diag along the correct axis
            shape = [1] * n_axes
            shape[axis] = len(c_diag)
            contracted = contracted * c_diag.reshape(shape)

        total = contracted.sum()
        return complex(r_val * total)

    def _evaluate_general(
        self,
        r_val: float,
        times: dict[str, float],
        directions: dict[str, Any],
        cache: PropagatorCache,
        idx_names: list[str],
        idx_name_to_axis: dict[str, int],
    ) -> complex:
        """General path: explicit loop over propagator index combinations."""
        spatial = self.spatial
        coeff = self.coupling_array
        dt = self.diagram_term
        prop_idx = dt.propagator_indices
        prop_shape = tuple(dim for _, dim in prop_idx)

        total = complex(0)
        for pidx in np.ndindex(*prop_shape):
            c_val = complex(coeff[pidx])
            if c_val == 0:
                continue

            idx_map = {name: val for name, val in zip(idx_names, pidx)}

            for sp_l, sp_r, il, ir in spatial.c_propagators:
                dir_l = spatial.direction_map[sp_l]
                dir_r = spatial.direction_map[sp_r]
                n_l = directions.get(dir_l, directions.get(sp_l))
                n_r = directions.get(dir_r, directions.get(sp_r))
                C_mat = cache.C_value(n_l, times[sp_l], n_r, times[sp_r])

                a = self._resolve_component(il, idx_map)
                b = self._resolve_component(ir, idx_map)
                if a is not None and b is not None:
                    c_val *= C_mat[a, b]
                else:
                    c_val *= C_mat.trace()

            # For non-iso R: include R matrix elements
            if not cache.model.iso_R:
                for sl, sr in spatial.r_propagators:
                    R_mat = cache.R_time(times[sl], times[sr])
                    # R propagators in the diagram may have indices
                    r_prop = self._find_r_propagator(sl, sr)
                    if r_prop and r_prop.index_left and r_prop.index_right:
                        a = self._resolve_component(r_prop.index_left, idx_map)
                        b = self._resolve_component(r_prop.index_right, idx_map)
                        if a is not None and b is not None:
                            c_val *= float(R_mat[a, b])
                        else:
                            c_val *= float(np.trace(R_mat))
                    # If iso_R with indices stripped, R is already in r_val
                r_val = 1.0  # already accounted for above

            total += c_val

        return r_val * total

    def _find_r_propagator(self, sl: str, sr: str) -> Propagator | None:
        """Find the R propagator matching spatial_left=sl, spatial_right=sr."""
        for p in self.diagram_term.propagators:
            if p.kind == "R" and p.spatial_left == sl and p.spatial_right == sr:
                return p
        return None

    @staticmethod
    def _resolve_component(
        idx_name: str | None, idx_map: dict[str, int]
    ) -> int | None:
        """Resolve a component index name to a 0-indexed integer."""
        if idx_name is None:
            return None
        if idx_name in idx_map:
            return idx_map[idx_name]
        try:
            return int(idx_name) - 1  # 1-indexed literal
        except ValueError:
            return None

    def make_scipy_integrand(
        self,
        external_times: dict[str, float],
        external_directions: dict[str, Any],
        cache: PropagatorCache,
    ) -> Callable:
        """Return a callable for ``scipy.integrate.nquad``.

        The returned function accepts the integration time variables as
        positional arguments in the order of
        ``spatial.time_integration_vars`` and returns the integrand value.
        External times and directions are baked in.

        Args:
            external_times: ``{point_name: time}`` for external spatial points.
            external_directions: ``{direction_var: value}`` for all independent
                direction variables.
            cache: A :class:`PropagatorCache`.

        Returns:
            Callable ``f(*time_args) → float``.
        """
        spatial = self.spatial
        int_vars = spatial.time_integration_vars

        def integrand(*time_args: float) -> float:
            times = dict(external_times)
            for var, val in zip(int_vars, time_args):
                times[var] = val
            result = self.evaluate(times, external_directions, cache)
            return result.real if result.imag == 0 else abs(result)

        return integrand

    def integration_bounds(
        self,
        external_times: dict[str, float],
        t_min: float = 0.0,
    ) -> list:
        """Return integration bounds for ``scipy.integrate.nquad``.

        Time ordering constraints from R causality translate to
        variable-dependent bounds.  For ``nquad``, bounds can be
        callables ``f(*earlier_args) → (lo, hi)``.

        The variables are ordered as ``spatial.time_integration_vars``.

        Returns:
            List of bounds, one per integration variable.  Each is either
            ``(lo, hi)`` or a callable for dependent bounds.
        """
        spatial = self.spatial
        int_vars = list(spatial.time_integration_vars)

        # Build map: for each integration var, what is its upper bound?
        # Upper bound = min(t_later) where t_later > t_var in causal order.
        # If t_later is external, it's a constant; if integration, it's a var.
        upper_bounds: dict[str, list[str]] = defaultdict(list)
        for earlier, later in spatial.time_orderings:
            if earlier in int_vars:
                upper_bounds[earlier].append(later)

        # Build bounds list
        bounds: list = []
        for i, var in enumerate(int_vars):
            ub_sources = upper_bounds.get(var, [])
            if not ub_sources:
                # No causal constraint — integrate from t_min to some max
                max_t = max(external_times.values()) if external_times else 1.0
                bounds.append((t_min, max_t))
            else:
                # Upper bound is min of all constraining times
                # For nquad, bounds[i] can be a function of the *previous* args
                # nquad calls bounds[i](*args[:i]) — but our vars may be in
                # different positions.  We need to handle this carefully.
                #
                # nquad convention: f(x0, x1, ..., x_{n-1}) where x0 is innermost.
                # bounds[i] is called with (x0, ..., x_{i-1}).
                # Our int_vars[0] is innermost (earliest time).
                _var = var
                _ub = ub_sources
                _ext = external_times
                _int_vars = int_vars
                _t_min = t_min

                def make_bound(
                    ub: list[str],
                    ext: dict[str, float],
                    ivars: list[str],
                    lo: float,
                ) -> Callable:
                    def bound_func(*prev_args: float) -> tuple[float, float]:
                        # prev_args correspond to int_vars[:current_index]
                        hi_vals: list[float] = []
                        for src in ub:
                            if src in ext:
                                hi_vals.append(ext[src])
                            else:
                                # Find position of src in int_vars
                                src_idx = ivars.index(src)
                                if src_idx < len(prev_args):
                                    hi_vals.append(prev_args[src_idx])
                                else:
                                    hi_vals.append(ext.get(src, 1.0))
                        hi = min(hi_vals) if hi_vals else 1.0
                        return (lo, hi)

                    return bound_func

                bounds.append(make_bound(_ub, _ext, _int_vars, _t_min))

        return bounds

    def to_latex(self) -> str:
        r"""Render the decomposed integrand with explicit coordinates.

        Shows direction identifications, time-ordered integrals,
        factored R product, and the component sum over C propagators.
        """
        from .expressions import _latex_index

        spatial = self.spatial
        dt = self.diagram_term
        parts: list[str] = []

        # --- Direction identification note ---
        for group in spatial.direction_groups:
            if len(group) > 1:
                rep = None
                for p in sorted(group):
                    if p in spatial.external_points:
                        rep = p
                        break
                if rep is None:
                    rep = sorted(group)[0]
                others = sorted(group - {rep})
                if others:
                    ids = ", ".join(
                        rf"\hat{{n}}_{{{_latex_index(o)}}}" for o in others
                    )
                    parts.append(
                        rf"\text{{[all directions}} = \hat{{n}}_{{{_latex_index(rep)}}}]"
                    )

        # --- Time-ordered integrals ---
        body_parts: list[str] = []

        # R product
        r_strs = []
        for sl, sr in spatial.r_propagators:
            r_strs.append(rf"R(t_{{{_latex_index(sl)}}}, t_{{{_latex_index(sr)}}})")
        if r_strs:
            body_parts.append(" ".join(r_strs))

        # Summation + C factors
        prop_idx = dt.propagator_indices
        sum_prefix = ""
        for name, dim in prop_idx:
            sum_prefix += rf"\sum_{{{_latex_index(name)}=1}}^{{{dim}}} "

        c_strs = []
        for sp_l, sp_r, il, ir in spatial.c_propagators:
            dir_l = spatial.direction_map[sp_l]
            dir_r = spatial.direction_map[sp_r]
            idx_str = ""
            if il is not None and ir is not None:
                idx_str = f"_{{{_latex_index(il)}{_latex_index(ir)}}}"
            c_strs.append(
                rf"C{idx_str}({dir_l}, t_{{{_latex_index(sp_l)}}};\, "
                rf"{dir_r}, t_{{{_latex_index(sp_r)}}})"
            )

        coeff_str = r"\mathrm{coeff}"
        if prop_idx:
            idx_sub = ",".join(_latex_index(n) for n, _ in prop_idx)
            coeff_str = rf"\mathrm{{coeff}}_{{{idx_sub}}}"

        inner = rf"{sum_prefix}{coeff_str} " + " ".join(c_strs)
        body_parts.append(inner)

        body = r" \cdot ".join(body_parts)

        # Wrap with time integrals
        for var in reversed(list(spatial.time_integration_vars)):
            body = rf"\int \mathrm{{d}}t_{{{_latex_index(var)}}}\, {body}"

        result = body
        if parts:
            result = " ".join(parts) + r" \quad " + body

        return result
