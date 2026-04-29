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
    # Optional white-noise component of the source-field correlation:
    #
    #   κ²(t1, x1; t2, x2) = kappa2_smooth + δ(t1 − t2) · sigma2(t1; x1, x2).
    #
    # ``sigma2`` takes a single time argument (the δ collapses both
    # times) and returns an (N, N) matrix.  When provided, ``C`` picks
    # up an additional 1-D integral
    #
    #   C_white(t1, t2; x1, x2) =
    #       ∫_{t_min}^{min(t1,t2)} R(t1, τ) σ²(τ; x1, x2) R(t2, τ) dτ,
    #
    # evaluated by :meth:`PropagatorCache._C_value_direct` and the
    # :meth:`~PropagatorCache.C_value` dblquad fallback alongside the
    # existing 2-D ``kappa2_smooth`` integral.  All
    # ``precompute_C_table_*`` builders call ``_C_value_direct``, so
    # the white-noise term is automatically absorbed into every
    # homogeneity-mode spline with no caller-side changes.
    sigma2: Callable | None = None


class _LazyTimeSplineCache:
    """Per-parameter-value 2-D ``(t1, t2)`` spline cache for lazy
    spatial evaluation.

    Backs the lazy mode of :class:`PropagatorCache` for each
    homogeneity symmetry: instead of pre-building an (n+1)-D spline
    over a pre-allocated parameter grid, this cache builds a 2-D
    time spline on demand for each distinct parameter value seen.

    Parameter semantics per mode:

    - ``translation``: key is ``r = |x1 − x2|`` (scalar ≥ 0).
    - ``rotation``: key is ``cos θ = x1·x2 / (|x1| |x2|)`` (scalar
      in ``[-1, 1]``).
    - ``general``: key is ``(x1_tuple, x2_tuple)`` — a pair of
      tuples over all x dimensions.

    Memoization key is the parameter rounded to ``round_decimals``
    decimal places so floating-point near-duplicates collapse.
    """

    def __init__(
        self,
        parent: "PropagatorCache",
        t_max: float,
        n_grid_t: int,
        mode: str,
        round_decimals: int = 10,
        n_jobs: int = 1,
    ):
        self.parent = parent
        self.t_max = t_max
        self.n_grid_t = n_grid_t
        self.mode = mode
        self.round_decimals = round_decimals
        self.ts = np.linspace(parent.model.t_min, t_max, n_grid_t)
        self._splines_by_key: dict = {}
        #: Parallel-worker count forwarded to :meth:`_build`.  ``1``
        #: (default) is serial; ``-1`` is all cores via
        #: :mod:`joblib.Parallel` with the ``loky`` backend.  Each
        #: on-demand build for a new parameter value pays a one-time
        #: worker-startup cost (~1 s) before the ``n_grid_t²``
        #: independent ``_C_value_direct`` calls fan out across
        #: cores.
        self.n_jobs = n_jobs

    def get_splines(self, x1_val, x2_val) -> list:
        """Return the list of per-component 2-D splines for this
        (x1, x2) pair.  Builds and memoizes on first call."""
        key = self._make_key(x1_val, x2_val)
        if key not in self._splines_by_key:
            self._splines_by_key[key] = self._build(x1_val, x2_val)
        return self._splines_by_key[key]

    def _make_key(self, x1_val, x2_val):
        """Derive the memoization key.  In translation/rotation mode
        the key is derived from the parameter (r or cos); in general
        mode it's the (x1, x2) tuple pair."""
        if self.mode == "translation":
            # Translation invariance: C depends only on ``||x1 - x2||``.
            # Accept scalar or arbitrary-dimensional vector inputs;
            # the cache shape stays (t1, t2, r) regardless of d.
            diff = np.asarray(x1_val, dtype=float) - np.asarray(x2_val, dtype=float)
            r = float(abs(diff)) if diff.ndim == 0 else float(np.linalg.norm(diff))
            return ("r", round(r, self.round_decimals))
        if self.mode == "rotation":
            cos_val = _rotation_cos(x1_val, x2_val)
            return ("cos", round(float(cos_val), self.round_decimals))
        # general: keep both endpoints
        a = tuple(np.round(np.atleast_1d(x1_val), self.round_decimals).tolist())
        b = tuple(np.round(np.atleast_1d(x2_val), self.round_decimals).tolist())
        return ("xx", a, b)

    def _build(self, x1_val, x2_val) -> list:
        """Build 2-D ``(t1, t2)`` splines at this specific parameter
        value by evaluating ``_C_value_direct`` on the time grid.

        Parallelises across the ``n_grid_t × n_grid_t`` grid points
        when :attr:`n_jobs` ≠ 1, mirroring the full-grid builders.
        """
        from scipy.interpolate import RectBivariateSpline

        parent = self.parent
        N = parent.model.n_components
        ts = self.ts
        n_t = self.n_grid_t

        tasks = [
            (i, j, ts[i], ts[j])
            for i in range(n_t)
            for j in range(n_t)
        ]

        x1_arr = np.asarray(x1_val)
        x2_arr = np.asarray(x2_val)

        def _point(args):
            i, j, t1, t2 = args
            C_mat = parent._C_value_direct(x1_arr, t1, x2_arr, t2)
            return i, j, np.array([C_mat[a, a] for a in range(N)])

        # Serial fast path; matches the full-grid builders' short-list
        # threshold so small caches don't pay joblib startup.
        if self.n_jobs == 1 or len(tasks) <= 4:
            results = [_point(t) for t in tasks]
        else:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(_point)(t) for t in tasks
            )

        grids = [np.zeros((n_t, n_t)) for _ in range(N)]
        for i, j, cvec in results:
            for a in range(N):
                grids[a][i, j] = cvec[a]

        return [RectBivariateSpline(ts, ts, grids[a]) for a in range(N)]


def _rotation_cos(x1, x2) -> float:
    """Cosine similarity ``x1·x2 / (|x1| |x2|)`` as a scalar.

    Works for scalar x (interpreted as 1-D vector) and array x.
    Returns ``1.0`` when either vector is exactly zero (degenerate;
    avoids a divide-by-zero NaN).
    """
    v1 = np.atleast_1d(np.asarray(x1, dtype=float))
    v2 = np.atleast_1d(np.asarray(x2, dtype=float))
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        return 1.0
    return float(np.dot(v1, v2) / (n1 * n2))


def _resolve_integrate_over(
    integrate_over: Any,
    ext_vars: list[str],
) -> set[str]:
    """Normalise the ``integrate_over`` kwarg into a set of external
    point names to integrate over.

    Accepts:

    - ``None`` — the empty set (all externals fixed at ``lambda_f``).
      This is the physics-observable convention
      ``⟨φ(t_f) · φ(t_f)⟩``.
    - ``"all"`` — the full set ``ext_vars`` (time-integrated moment).
    - Iterable of names — used as-is, after validating that each
      name appears in ``ext_vars``.

    Raises ``ValueError`` on an unknown external name.
    """
    if integrate_over is None:
        return set()
    if isinstance(integrate_over, str):
        if integrate_over == "all":
            return set(ext_vars)
        raise ValueError(
            f"integrate_over must be None, 'all', or an iterable of "
            f"external-point names; got string {integrate_over!r}."
        )
    requested = set(integrate_over)
    unknown = requested - set(ext_vars)
    if unknown:
        raise ValueError(
            f"integrate_over references external points that don't "
            f"exist in this diagram: {sorted(unknown)}.  Known "
            f"externals: {ext_vars}."
        )
    return requested


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
        homogeneity: str = "translation",
        interp_method: str = "linear",
    ):
        """Initialise the propagator cache.

        Args:
            model: Propagator model specifying R and κ².
            quad_opts: Options forwarded to ``scipy.integrate.dblquad``.
            homogeneity: Physical assumption about how C depends on the
                spatial coordinates.  Determines which
                ``precompute_C_table_*`` builder is valid.

                - ``'translation'`` (default): C depends only on
                  ``|x1 − x2|``.  Appropriate for translation-invariant
                  (spatially stationary) noise — the most common case.
                  Build via :meth:`precompute_C_table_translation`.
                - ``'rotation'``: C depends only on ``x1 · x2``.
                  Appropriate when x is a unit direction vector (e.g.
                  on the sphere).  Build via
                  :meth:`precompute_C_table_rotation`.
                - ``'general'``: no symmetry assumed; C is evaluated on
                  a full 4-D grid in ``(t1, t2, x1, x2)``.  Build via
                  :meth:`precompute_C_table_general`.

                When a spatial builder is called with its ``*_grid``
                argument left ``None``, the cache enters **lazy mode**
                for that symmetry: 2-D ``(t1, t2)`` splines are built
                on-demand for each distinct parameter value queried,
                and memoized.  This is the right default for
                moment-at-fixed-points workflows where only a few
                distinct parameter values ever appear.
            interp_method: ``RegularGridInterpolator`` method used by
                the three full-grid spline builders
                (``precompute_C_table_translation``,
                ``precompute_C_table_rotation``,
                ``precompute_C_table_general``).

                - ``'linear'`` (default, safe): O(h²) accuracy,
                  monotone -- never overshoots / flips sign even on
                  steeply-decaying C tails. Required when C spans
                  many orders of magnitude across the r-grid (the
                  typical cosmological setting; see
                  ``tests/test_evaluate_interpolation_accuracy.py``).
                - ``'cubic'``: O(h⁴) accuracy on smooth, well-sampled
                  grids -- only safe when the user knows their C is
                  bounded away from grid-induced sign-flip artefacts.

                ``RegularGridInterpolator`` accepts any other method
                name supported by the installed scipy (e.g.
                ``'quintic'``, ``'pchip'``, ``'nearest'``); they are
                forwarded as-is and validated lazily on table build.
        """
        if homogeneity not in ("translation", "rotation", "general"):
            raise ValueError(
                f"homogeneity must be 'translation', 'rotation', or "
                f"'general'; got {homogeneity!r}"
            )
        self.model = model
        self.quad_opts = quad_opts or {}
        self.homogeneity = homogeneity
        self.interp_method = interp_method

        # Closed-form-only path: skip every spline and route C lookups
        # directly through ``self._C_value_direct``. Set by
        # ``Propagators.build()`` when both ``c_closed_form`` and
        # ``c_closed_form_only=True`` are passed; the user's analytical
        # C function then becomes the lookup function with no
        # interpolation error. ``closed_form_vectorized`` flags whether
        # the user's c_fn accepts batched (n,)-shape t and (n,)- or
        # broadcast-scalar n inputs and returns ``(n, N, N)``; if not,
        # the per-sample fallback is used (slow, but correct).
        self._closed_form_only: bool = False
        self._closed_form_vectorized: bool = False

        self._c_cache: dict[tuple, np.ndarray] = {}
        # Legacy 2-D (t1, t2) spline at fixed x — still populated by
        # ``precompute_C_table`` for backward compatibility.
        self._c_splines: list | None = None
        self._c_table_range: tuple[float, float] | None = None

        # Full N-D spatial splines, populated by precompute_C_table_*
        # with an explicit parameter grid:
        self._c_translation_splines: list | None = None  # 3-D (t1,t2,r)
        self._c_translation_r_range: tuple[float, float] | None = None
        self._c_rotation_splines: list | None = None    # 3-D (t1,t2,cos)
        self._c_general_interpolators: list | None = None  # 4-D
        self._c_general_axes: tuple | None = None

        # Lazy per-parameter-value 2-D (t1, t2) spline caches.  A
        # :class:`_LazyTimeSplineCache` is attached when the user
        # calls a ``precompute_C_table_*`` method without a parameter
        # grid (see ``lazy mode`` in the docstring).  Keyed by
        # rounded parameter value (r, cos, or (x1, x2)).
        self._lazy_translation: _LazyTimeSplineCache | None = None
        self._lazy_rotation: _LazyTimeSplineCache | None = None
        self._lazy_general: _LazyTimeSplineCache | None = None

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

        If :meth:`precompute_C_table` has been called and ``t1``, ``t2``
        are within the table range, uses fast spline interpolation.
        Otherwise falls back to ``scipy.integrate.dblquad``::

            C_{ab} = ∫_{t_min}^{t1} dλ' ∫_{t_min}^{t2} dλ''
                     R_time(t1,λ') κ_{ab}(n1,λ'; n2,λ'') R_time(t2,λ'')

        Returns:
            ``(N, N)`` array.
        """
        # Fast path (spatial-aware): route through C_at_batch which
        # dispatches on ``homogeneity`` + full/lazy table presence.
        # Only used when a spatial table has actually been built —
        # otherwise fall through to the legacy/dblquad paths so the
        # plain ``PropagatorCache(model)`` + legacy ``precompute_C_table``
        # flow stays bit-identical to before.
        has_spatial_table = (
            self._c_translation_splines is not None
            or self._lazy_translation is not None
            or self._c_rotation_splines is not None
            or self._lazy_rotation is not None
            or self._c_general_interpolators is not None
            or self._lazy_general is not None
        )
        if has_spatial_table and self.model.diag_C:
            N = self.model.n_components
            t1_arr = np.array([t1], dtype=float)
            t2_arr = np.array([t2], dtype=float)
            # n1, n2 may be scalars or arrays (e.g. np.float64 or
            # small np.ndarray); pass through np.asarray so the
            # downstream rotation/general paths see consistent shape.
            x1_arr = np.asarray(n1, dtype=float).reshape(1, -1) \
                if np.ndim(n1) > 0 else np.array([float(n1)])
            x2_arr = np.asarray(n2, dtype=float).reshape(1, -1) \
                if np.ndim(n2) > 0 else np.array([float(n2)])
            if x1_arr.ndim == 2 and x1_arr.shape[-1] == 1:
                x1_arr = x1_arr.ravel()
                x2_arr = x2_arr.ravel()
            c_diag = self.C_at_batch(t1_arr, t2_arr, x1_arr, x2_arr)[0]
            C_mat = np.zeros((N, N))
            for a in range(N):
                C_mat[a, a] = c_diag[a]
            return C_mat

        # Fast path (legacy time-only): 2-D spline table
        if self._c_splines is not None and self.model.diag_C:
            lo, hi = self._c_table_range  # type: ignore[misc]
            if lo <= t1 <= hi and lo <= t2 <= hi:
                return self._C_value_from_table(t1, t2)

        # Check (n1, t1, n2, t2) LRU cache
        cache_key = (id(n1) if isinstance(n1, np.ndarray) else n1,
                     t1, id(n2) if isinstance(n2, np.ndarray) else n2, t2)
        if cache_key in self._c_cache:
            return self._c_cache[cache_key]

        # Delegate to ``_C_value_direct`` (which adds the white-noise
        # 1-D contribution when ``model.sigma2`` is set), then store
        # in the per-arg LRU.
        C_mat = self._C_value_direct(n1, t1, n2, t2)
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
        Uses spline table when available for fast lookup.
        """
        if n_prime is None:
            n_prime = n
        if t2 is None:
            t2 = t1
        # Fast path: spline table
        if self._c_splines is not None:
            lo, hi = self._c_table_range  # type: ignore[misc]
            if lo <= t1 <= hi and lo <= t2 <= hi:
                return self._C_diagonal_from_table(t1, t2)
        C_mat = self.C_value(n, t1, n_prime, t2)
        return np.diag(C_mat)

    def clear_cache(self) -> None:
        """Clear the C value cache and spline table."""
        self._c_cache.clear()
        self._c_splines = None
        self._c_table_range = None

    def precompute_C_table(
        self,
        t_max: float,
        n_grid: int = 100,
        direction: Any = 0,
    ) -> None:
        """Pre-compute C on a grid and build spline interpolators.

        For ``iso_R + diag_C`` with direction-independent ``kappa2``,
        ``C_{aa}(t1, t2)`` depends only on ``(t1, t2)``.  This method
        evaluates C on an ``n_grid × n_grid`` grid via ``dblquad``,
        then builds a :class:`~scipy.interpolate.RectBivariateSpline`
        for each diagonal component.

        After calling this, :meth:`C_value` and :meth:`C_diagonal`
        use fast spline interpolation instead of ``dblquad``.

        Args:
            t_max: Upper bound of the time grid.
            n_grid: Number of grid points per axis (default 100).
            direction: Direction value to use for kappa2 evaluation.
        """
        from scipy.interpolate import RectBivariateSpline

        m = self.model
        N = m.n_components
        t_min = m.t_min
        ts = np.linspace(t_min, t_max, n_grid)

        # Build grid for each diagonal component
        grids = [np.zeros((n_grid, n_grid)) for _ in range(N)]

        for i in range(n_grid):
            for j in range(n_grid):
                C_mat = self.C_value(direction, ts[i], direction, ts[j])
                for a in range(N):
                    grids[a][i, j] = C_mat[a, a]

        self._c_splines = [
            RectBivariateSpline(ts, ts, grids[a])
            for a in range(N)
        ]
        self._c_table_range = (t_min, t_max)

    @property
    def has_table(self) -> bool:
        """Whether a pre-computed C table is available."""
        return self._c_splines is not None

    def _C_diagonal_from_table(self, t1: float, t2: float) -> np.ndarray:
        """Look up C diagonal from spline table."""
        return np.array([
            float(s(t1, t2, grid=False)) for s in self._c_splines  # type: ignore[union-attr]
        ])

    def _C_value_from_table(self, t1: float, t2: float) -> np.ndarray:
        """Look up C matrix from spline table (diagonal only)."""
        N = self.model.n_components
        C_mat = np.zeros((N, N))
        for a, s in enumerate(self._c_splines):  # type: ignore[union-attr]
            C_mat[a, a] = float(s(t1, t2, grid=False))
        return C_mat

    # --- Vectorized methods for batch evaluation ---

    def C_diagonal_batch(
        self, t1: np.ndarray, t2: np.ndarray,
    ) -> np.ndarray:
        """Evaluate diagonal C at arrays of time pairs.

        Requires :meth:`precompute_C_table` to have been called.

        Args:
            t1: Array of shape ``(n,)`` — first time coordinates.
            t2: Array of shape ``(n,)`` — second time coordinates.

        Returns:
            Array of shape ``(n, N)`` where ``N`` is the number of
            field components.  ``result[s, a]`` is ``C_{aa}(t1[s], t2[s])``.
        """
        if self._c_splines is None:
            raise RuntimeError(
                "C_diagonal_batch requires precompute_C_table()"
            )
        n = len(t1)
        N = self.model.n_components
        result = np.empty((n, N))
        for a, s in enumerate(self._c_splines):
            result[:, a] = s(t1, t2, grid=False)
        return result

    def R_time_batch(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        """Evaluate R_time at arrays of time pairs (iso_R only).

        Args:
            t1: Array of shape ``(n,)`` — left times.
            t2: Array of shape ``(n,)`` — right times.

        Returns:
            Array of shape ``(n,)``.  R(t1, t2) = Θ(t1−t2) × R_time(t1, t2).
        """
        # Vectorize the user-provided R_time function
        R_vec = np.vectorize(self.model.R_time)
        return R_vec(t1, t2)

    # ------------------------------------------------------------------ #
    # Spatial-aware extension: (t, x) coordinates via homogeneity modes
    # ------------------------------------------------------------------ #

    def precompute_C_table_translation(
        self,
        t_max: float,
        n_grid_t: int = 60,
        r_max: float | None = None,
        n_grid_r: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        """Enable spatial support under the translation-invariance
        assumption:

            ``C(t1, t2; x1, x2)`` depends only on ``|x1 − x2|``.

        Two operating modes depending on whether ``r_max`` and
        ``n_grid_r`` are supplied:

        **Lazy mode (default, recommended for fixed-point moments)**
            ``r_max=None`` and ``n_grid_r=None``.  No r-grid is
            pre-computed.  When :meth:`C_at_batch` encounters a new
            ``r = |x1−x2|`` value it builds and caches a 2-D
            ``(t1, t2)`` spline on demand.  For a moment calculation
            at fixed external points ``(x_1, …, x_n)`` there are at
            most ``O(n²)`` distinct r-values, so lazy mode saves an
            enormous amount of work vs a densely-sampled r-grid —
            particularly when the underlying ``_C_value_direct`` is
            ``scipy.integrate.dblquad`` (expensive).

        **Full-grid mode (right for sweeping r curves)**
            Both ``r_max`` and ``n_grid_r`` provided.  A 3-D
            ``(t1, t2, r)`` spline is built up-front over the full
            range, giving O(1) evaluation per query — faster once
            ``n_distinct_r >> n_grid_r``.  Appropriate when the user
            will plot ``C`` or a derived observable as a function of
            r over a continuous range.

        Args:
            t_max: upper bound of the (t1, t2) time grid.
            n_grid_t: grid size along each time axis.
            r_max: upper bound of the r = |x1-x2| grid
                (full-grid mode).  ``None`` ⇒ lazy mode.
            n_grid_r: grid size along r (full-grid mode).
                ``None`` ⇒ lazy mode.
            n_jobs: parallel workers for grid-point ``_C_value_direct``
                evaluations.  ``1`` serial, ``-1`` all cores via
                :mod:`joblib` (``loky`` backend).  Applies in **both**
                full-grid mode (fans out across the 3-D ``(t1, t2, r)``
                grid) **and** lazy mode (fans out across each on-demand
                2-D ``(t1, t2)`` build; each new r pays a ~1 s worker
                startup cost).

        Requires ``self.homogeneity == 'translation'``.

        Side effect: sets either ``_c_translation_splines`` (full)
        or ``_lazy_translation`` (lazy).
        """
        if self.homogeneity != "translation":
            raise RuntimeError(
                f"precompute_C_table_translation requires "
                f"homogeneity='translation', got {self.homogeneity!r}"
            )

        m = self.model
        t_min = m.t_min
        self._c_table_range = (t_min, t_max)

        # Lazy mode: defer per-r spline construction to first query.
        if r_max is None or n_grid_r is None:
            self._lazy_translation = _LazyTimeSplineCache(
                self, t_max, n_grid_t, mode="translation",
                n_jobs=n_jobs,
            )
            return

        # Full-grid mode: build the 3-D (t1, t2, r) spline now.
        from scipy.interpolate import RegularGridInterpolator

        N = m.n_components
        ts = np.linspace(t_min, t_max, n_grid_t)
        rs = np.linspace(0.0, r_max, n_grid_r)

        tasks = [
            (i, j, k, ts[i], ts[j], rs[k])
            for i in range(n_grid_t)
            for j in range(n_grid_t)
            for k in range(n_grid_r)
        ]

        def _point(args):
            i, j, k, t1, t2, r = args
            # TODO(d-dim): replace scalar r with np.ndarray x_diff
            C_mat = self._C_value_direct(
                np.asarray(0.0), t1, np.asarray(r), t2,
            )
            return i, j, k, np.array([C_mat[a, a] for a in range(N)])

        if n_jobs == 1 or len(tasks) <= 4:
            results = [_point(t) for t in tasks]
        else:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_point)(t) for t in tasks
            )

        grids = [np.zeros((n_grid_t, n_grid_t, n_grid_r)) for _ in range(N)]
        for i, j, k, cvec in results:
            for a in range(N):
                grids[a][i, j, k] = cvec[a]

        self._c_translation_splines = [
            RegularGridInterpolator(
                (ts, ts, rs), grids[a],
                bounds_error=False, fill_value=None,
                # Default 'linear' (set in PropagatorCache.__init__):
                # tensor-product cubic on a steeply decaying C produces
                # sign flips and O(1) overshoot in the tail; linear is
                # monotone and safe. Users with smooth C and dense
                # grids can opt into cubic via interp_method='cubic'.
                # See tests/test_evaluate_interpolation_accuracy.py.
                method=self.interp_method,
            )
            for a in range(N)
        ]
        self._c_translation_r_range = (0.0, r_max)

    def precompute_C_table_rotation(
        self,
        t_max: float,
        n_grid_t: int = 60,
        n_grid_cos: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        """Enable spatial support under the rotation-invariance
        assumption:

            ``C(t1, t2; x1, x2)`` depends only on ``x1 · x2``.

        Appropriate when ``x`` is a unit direction vector (e.g. on
        the sphere).  The effective spatial parameter is
        ``cos θ ∈ [−1, 1]``.

        Two modes, analogous to
        :meth:`precompute_C_table_translation`:

        - Lazy (``n_grid_cos=None``): 2-D ``(t1, t2)`` splines built
          on demand per distinct ``cos θ`` value.  Best when only
          a few ``cos θ`` values arise (e.g. a fixed-set moment
          calculation).
        - Full-grid (``n_grid_cos`` integer): 3-D
          ``(t1, t2, cos)`` spline over the whole ``cos ∈ [−1, 1]``
          range.  Best for angular-correlation sweeps.

        Under the hood, the reference evaluation at a specific
        ``cos θ`` uses two representative unit vectors
        ``e_1 = (1, 0, …)`` and ``e_2`` chosen so that
        ``e_1 · e_2 = cos θ`` (a 2-D rotation of ``e_1``).

        Args:
            n_jobs: parallel workers for grid-point
                ``_C_value_direct`` evaluations; applies in **both**
                full-grid (fans out across the 3-D ``(t1, t2, cos)``
                grid) and lazy mode (fans out across each on-demand
                2-D build).  ``1`` serial, ``-1`` all cores via
                :mod:`joblib`.

        Requires ``self.homogeneity == 'rotation'``.
        """
        if self.homogeneity != "rotation":
            raise RuntimeError(
                f"precompute_C_table_rotation requires "
                f"homogeneity='rotation', got {self.homogeneity!r}"
            )

        m = self.model
        t_min = m.t_min
        self._c_table_range = (t_min, t_max)

        if n_grid_cos is None:
            self._lazy_rotation = _LazyTimeSplineCache(
                self, t_max, n_grid_t, mode="rotation",
                n_jobs=n_jobs,
            )
            return

        from scipy.interpolate import RegularGridInterpolator

        N = m.n_components
        ts = np.linspace(t_min, t_max, n_grid_t)
        coses = np.linspace(-1.0, 1.0, n_grid_cos)

        def _rep_vectors(cos_val):
            """Return a pair of unit vectors with dot product cos_val.

            We use the 2-D representation ``e_1 = (1, 0)`` and
            ``e_2 = (cos, sin)`` where ``sin = √(1 − cos²)``.  This
            is sufficient when ``kappa2`` depends only on ``x1·x2``
            (rotation invariance makes the ambient dimension
            irrelevant).
            """
            c = float(cos_val)
            c = max(-1.0, min(1.0, c))
            s = float(np.sqrt(max(0.0, 1.0 - c * c)))
            return np.array([1.0, 0.0]), np.array([c, s])

        tasks = [
            (i, j, k, ts[i], ts[j], coses[k])
            for i in range(n_grid_t)
            for j in range(n_grid_t)
            for k in range(n_grid_cos)
        ]

        def _point(args):
            i, j, k, t1, t2, cos_val = args
            e1, e2 = _rep_vectors(cos_val)
            C_mat = self._C_value_direct(e1, t1, e2, t2)
            return i, j, k, np.array([C_mat[a, a] for a in range(N)])

        if n_jobs == 1 or len(tasks) <= 4:
            results = [_point(t) for t in tasks]
        else:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_point)(t) for t in tasks
            )

        grids = [np.zeros((n_grid_t, n_grid_t, n_grid_cos)) for _ in range(N)]
        for i, j, k, cvec in results:
            for a in range(N):
                grids[a][i, j, k] = cvec[a]

        self._c_rotation_splines = [
            RegularGridInterpolator(
                (ts, ts, coses), grids[a],
                bounds_error=False, fill_value=None,
                # Default 'linear' (set in PropagatorCache.__init__):
                # tensor-product cubic on a steeply decaying C produces
                # sign flips and O(1) overshoot in the tail; linear is
                # monotone and safe. Users with smooth C and dense
                # grids can opt into cubic via interp_method='cubic'.
                # See tests/test_evaluate_interpolation_accuracy.py.
                method=self.interp_method,
            )
            for a in range(N)
        ]

    def precompute_C_table_general(
        self,
        t_max: float,
        n_grid_t: int = 40,
        x_max: float | None = None,
        n_grid_x: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        """Pre-compute C with no spatial symmetry assumed.

        Two modes:

        - Lazy (``x_max=None`` or ``n_grid_x=None``): 2-D
          ``(t1, t2)`` splines built on demand per distinct
          ``(x1, x2)`` pair.  Best for fixed-point moments where
          only a handful of pairs occur.
        - Full-grid: 4-D ``(t1, t2, x1, x2)`` interpolator over the
          pre-allocated grid.  Build cost is
          ``n_grid_t² · n_grid_x²`` independent ``_C_value_direct``
          calls — strongly consider ``n_jobs=-1``.

        Args:
            n_jobs: parallel workers for ``_C_value_direct``
                evaluations; applies in **both** full-grid (fans out
                across the 4-D grid) and lazy mode (fans out across
                each on-demand 2-D build).  ``1`` serial, ``-1`` all
                cores via :mod:`joblib`.

        Requires ``self.homogeneity == 'general'``.
        """
        if self.homogeneity != "general":
            raise RuntimeError(
                f"precompute_C_table_general requires "
                f"homogeneity='general', got {self.homogeneity!r}"
            )

        m = self.model
        t_min = m.t_min
        self._c_table_range = (t_min, t_max)

        if x_max is None or n_grid_x is None:
            self._lazy_general = _LazyTimeSplineCache(
                self, t_max, n_grid_t, mode="general",
                n_jobs=n_jobs,
            )
            return

        from scipy.interpolate import RegularGridInterpolator

        N = m.n_components
        ts = np.linspace(t_min, t_max, n_grid_t)
        xs = np.linspace(-x_max, x_max, n_grid_x)

        tasks = [
            (i, j, p, q, ts[i], ts[j], xs[p], xs[q])
            for i in range(n_grid_t)
            for j in range(n_grid_t)
            for p in range(n_grid_x)
            for q in range(n_grid_x)
        ]

        def _point(args):
            i, j, p, q, t1, t2, x1, x2 = args
            # TODO(d-dim): x1, x2 would be vectors
            C_mat = self._C_value_direct(
                np.asarray(x1), t1, np.asarray(x2), t2,
            )
            return i, j, p, q, np.array([C_mat[a, a] for a in range(N)])

        if n_jobs == 1 or len(tasks) <= 4:
            results = [_point(t) for t in tasks]
        else:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_point)(t) for t in tasks
            )

        grids = [
            np.zeros((n_grid_t, n_grid_t, n_grid_x, n_grid_x))
            for _ in range(N)
        ]
        for i, j, p, q, cvec in results:
            for a in range(N):
                grids[a][i, j, p, q] = cvec[a]

        self._c_general_interpolators = [
            RegularGridInterpolator(
                (ts, ts, xs, xs), grids[a],
                bounds_error=False, fill_value=None,
                # Default 'linear' (set in PropagatorCache.__init__):
                # tensor-product cubic on a steeply decaying C produces
                # sign flips and O(1) overshoot in the tail; linear is
                # monotone and safe. Users with smooth C and dense
                # grids can opt into cubic via interp_method='cubic'.
                # See tests/test_evaluate_interpolation_accuracy.py.
                method=self.interp_method,
            )
            for a in range(N)
        ]
        self._c_general_axes = (ts, ts, xs, xs)

    def _C_value_direct(
        self,
        n1: Any,
        t1: float,
        n2: Any,
        t2: float,
    ) -> np.ndarray:
        """Direct quadrature evaluation of C without consulting any cache.

        Needed so the ``precompute_C_table_{translation,rotation,general}``
        builders aren't contaminated by the spline cache they are
        themselves populating.

        When ``model.sigma2`` is provided, the full κ² is
        ``κ²_smooth + δ(t1−t2) · σ²(t; x1, x2)``.  The δ collapses one
        time integral so C picks up an extra 1-D piece::

            C_white[a,b] = ∫_{t_min}^{min(t1,t2)} R(t1,τ) σ²[a,b](τ;
                            n1, n2) R(t2,τ) dτ

        which is added to the usual 2-D dblquad result.
        """
        from scipy.integrate import dblquad, quad as _quad

        m = self.model
        N = m.n_components
        t_min = m.t_min
        C_mat = np.zeros((N, N))

        if m.diag_C:
            for a in range(N):
                def integrand(lam2: float, lam1: float, _a: int = a) -> float:
                    r1 = float(m.R_time(t1, lam1)) if m.iso_R else float(m.R_time(t1, lam1)[_a, _a])
                    kappa_mat = m.kappa2(n1, lam1, n2, lam2)
                    r2 = float(m.R_time(t2, lam2)) if m.iso_R else float(m.R_time(t2, lam2)[_a, _a])
                    return r1 * float(kappa_mat[_a, _a]) * r2
                val, _ = dblquad(
                    integrand, t_min, t1, t_min, t2, **self.quad_opts,
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
                        integrand, t_min, t1, t_min, t2, **self.quad_opts,
                    )
                    C_mat[a, b] = val

        # --- White-noise (δ-correlated) component: 1-D integral ---
        if m.sigma2 is not None:
            t_upper = min(t1, t2)
            if t_upper > t_min:
                if m.diag_C:
                    for a in range(N):
                        def w_integrand(tau: float, _a: int = a) -> float:
                            r1 = float(m.R_time(t1, tau)) if m.iso_R \
                                else float(m.R_time(t1, tau)[_a, _a])
                            r2 = float(m.R_time(t2, tau)) if m.iso_R \
                                else float(m.R_time(t2, tau)[_a, _a])
                            sig = m.sigma2(n1, tau, n2)
                            return r1 * float(sig[_a, _a]) * r2
                        wval, _ = _quad(
                            w_integrand, t_min, t_upper, **self.quad_opts,
                        )
                        C_mat[a, a] += wval
                else:
                    for a in range(N):
                        for b in range(N):
                            def w_integrand(tau: float, _a: int = a, _b: int = b) -> float:
                                if m.iso_R:
                                    r1 = float(m.R_time(t1, tau))
                                    r2 = float(m.R_time(t2, tau))
                                else:
                                    r1 = float(m.R_time(t1, tau)[_a, _a])
                                    r2 = float(m.R_time(t2, tau)[_b, _b])
                                sig = m.sigma2(n1, tau, n2)
                                return r1 * float(sig[_a, _b]) * r2
                            wval, _ = _quad(
                                w_integrand, t_min, t_upper, **self.quad_opts,
                            )
                            C_mat[a, b] += wval

        return C_mat

    def C_at_batch(
        self,
        t1: np.ndarray,
        t2: np.ndarray,
        x1: np.ndarray | float,
        x2: np.ndarray | float,
    ) -> np.ndarray:
        """Batch-evaluate C at arbitrary ``(t1, t2, x1, x2)`` arrays.

        Dispatches on :attr:`homogeneity`:

        - ``'translation'``: C depends only on ``|x1 − x2|``.  Uses
          the 3-D ``(t1, t2, r)`` full-grid spline if built via
          ``precompute_C_table_translation(r_max=..., n_grid_r=...)``;
          otherwise uses a lazy per-r 2-D spline cache; otherwise
          (legacy 2-D spline only) falls back to
          :meth:`C_diagonal_batch` — x is ignored in that case.
        - ``'rotation'``: C depends only on ``x1 · x2 / (|x1| |x2|)``.
          Uses the 3-D ``(t1, t2, cos)`` spline or per-cos lazy cache.
        - ``'general'``: 4-D ``(t1, t2, x1, x2)`` spline, or a lazy
          per-pair 2-D cache.

        Args:
            t1, t2: Time arrays of shape ``(n,)``.
            x1, x2: Spatial coordinates.  Scalar or array of shape
                ``(n,)`` (broadcast if scalar).
                # TODO(d-dim): support (n, d).

        Returns:
            Array of shape ``(n, N)`` — diagonal C values per
            component.
        """
        t1 = np.atleast_1d(np.asarray(t1))
        t2 = np.atleast_1d(np.asarray(t2))
        N = self.model.n_components

        # Closed-form-only: skip every spline and call the user's
        # analytical C function directly. Used by
        # ``Propagators(c_closed_form=..., c_closed_form_only=True)``;
        # gives machine-precision agreement with the analytical C
        # (no interpolation error). The vectorised contract is the
        # fast path; the per-sample fallback runs a Python loop and
        # is only practical for small sweeps.
        if self._closed_form_only:
            return self._closed_form_at_batch_diag(t1, t2, x1, x2)
        n = len(t1)

        def _resolve_x(x):
            """Accept scalar, (n,) array, (d,) vector, or (n, d)
            array.  Return shape (n,) for 1-D x, (n, d) for d-D x.
            """
            arr = np.asarray(x, dtype=float)
            if arr.ndim == 0:
                return np.full(n, float(arr))
            if arr.ndim == 1:
                if arr.shape[0] == n:
                    return arr  # (n,) scalar-per-sample
                # treat as (d,) vector → broadcast to (n, d)
                return np.tile(arr[None, :], (n, 1))
            return arr  # (n, d) already

        x1_b = _resolve_x(x1)
        x2_b = _resolve_x(x2)

        def _r_batch(a, b):
            """``|a - b|`` (scalar x) or ``‖a - b‖`` (vector x)."""
            diff = a - b
            if diff.ndim == 1:
                return np.abs(diff)
            return np.linalg.norm(diff, axis=-1)

        if self.homogeneity == "translation":
            # Full-grid 3-D (t1, t2, r) spline takes precedence.
            if self._c_translation_splines is not None:
                r = _r_batch(x1_b, x2_b)
                pts = np.stack([t1, t2, r], axis=-1)
                result = np.empty((n, N))
                for a, itp in enumerate(self._c_translation_splines):
                    result[:, a] = itp(pts)
                return result
            if self._lazy_translation is not None:
                return self._lazy_lookup(
                    self._lazy_translation, t1, t2, x1_b, x2_b, N,
                )
            # Fall back to legacy 2-D (t1, t2) spline — x is
            # effectively ignored (r=0 assumed).  Matches behaviour
            # of users who only called the legacy
            # :meth:`precompute_C_table`.
            return self.C_diagonal_batch(t1, t2)

        if self.homogeneity == "rotation":
            if self._c_rotation_splines is not None:
                # cos per sample
                cosv = np.empty(n)
                for k in range(n):
                    cosv[k] = _rotation_cos(x1_b[k], x2_b[k])
                pts = np.stack([t1, t2, cosv], axis=-1)
                result = np.empty((n, N))
                for a, itp in enumerate(self._c_rotation_splines):
                    result[:, a] = itp(pts)
                return result
            if self._lazy_rotation is not None:
                return self._lazy_lookup(
                    self._lazy_rotation, t1, t2, x1_b, x2_b, N,
                )
            raise RuntimeError(
                "homogeneity='rotation' but no rotation table has "
                "been built — call precompute_C_table_rotation() first"
            )

        # homogeneity == "general"
        if self._c_general_interpolators is not None:
            # Full-grid general mode is built over a 1-D x axis
            # (precompute_C_table_general uses ``np.linspace(-x_max,
            # x_max, n_grid_x)``). Extending the grid to d dimensions
            # would yield a (2 + 2d)-D spline whose build cost scales
            # as ``n_grid_x ** (2d)`` -- exponentially expensive even
            # for d=2. Reject vector inputs in full-grid mode and
            # direct the user to lazy mode, which already supports
            # d-dim via dict-keyed memoisation.
            if x1_b.ndim > 1 or x2_b.ndim > 1:
                raise NotImplementedError(
                    "general-mode full-grid C tables only support "
                    "scalar (1-D) spatial coordinates because the "
                    "spline grid scales as n_grid_x**(2d). Use lazy "
                    "mode (omit x_max / n_grid_x in "
                    "precompute_C_table_general) for d-dim inputs."
                )
            pts = np.stack([t1, t2, x1_b, x2_b], axis=-1)
            result = np.empty((n, N))
            for a, itp in enumerate(self._c_general_interpolators):
                result[:, a] = itp(pts)
            return result
        if self._lazy_general is not None:
            return self._lazy_lookup(
                self._lazy_general, t1, t2, x1_b, x2_b, N,
            )
        raise RuntimeError(
            "homogeneity='general' but no general table has been "
            "built — call precompute_C_table_general() first"
        )

    def _closed_form_at_batch_diag(
        self,
        t1: np.ndarray,
        t2: np.ndarray,
        x1,
        x2,
    ) -> np.ndarray:
        """Direct lookup for ``closed_form_only=True``: skips every
        spline, calls ``self._C_value_direct`` (the user's c_fn) and
        returns the per-sample diagonal of the (N, N) result.

        Two contracts:

        * **Vectorised** (``self._closed_form_vectorized = True``):
          single call ``c_fn(x1, t1, x2, t2) -> (n, N, N)``.  Recommended
          for sweep performance.
        * **Per-sample** (default): Python loop calling
          ``c_fn(x1_i, t1_i, x2_i, t2_i) -> (N, N)`` per sample.
          Always correct, but ~50-100x slower than the vectorised
          path on typical workloads -- only practical for small
          point evaluations or testing.
        """
        N = self.model.n_components
        n = len(t1)

        # Resolve x1 / x2 to per-sample arrays so the loop body can
        # always slice ``x1_b[i]``. Accepts scalar, (n,), or (n, d).
        def _broadcast_x(x):
            arr = np.asarray(x, dtype=float)
            if arr.ndim == 0:
                return np.full(n, float(arr))
            if arr.ndim == 1 and arr.shape[0] == n:
                return arr
            if arr.ndim == 1:
                # (d,) vector → broadcast to (n, d)
                return np.tile(arr[None, :], (n, 1))
            return arr  # (n, d) already

        x1_b = _broadcast_x(x1)
        x2_b = _broadcast_x(x2)

        if self._closed_form_vectorized:
            full = np.asarray(self._C_value_direct(x1_b, t1, x2_b, t2))
            if full.ndim != 3 or full.shape[0] != n:
                raise ValueError(
                    f"vectorised closed-form c_fn must return shape "
                    f"(n, N, N); got {full.shape} for n={n}, N={N}."
                )
            # Diagonal per sample: result[i, a] = full[i, a, a].
            return np.einsum("iaa->ia", full).astype(float, copy=False)

        # Per-sample fallback. Slow but always correct.
        result = np.empty((n, N), dtype=float)
        for i in range(n):
            xi = x1_b[i]
            xj = x2_b[i]
            C_mat = np.asarray(self._C_value_direct(xi, t1[i], xj, t2[i]))
            result[i] = np.diag(C_mat)
        return result

    def _lazy_lookup(
        self,
        lazy: "_LazyTimeSplineCache",
        t1: np.ndarray,
        t2: np.ndarray,
        x1: np.ndarray,
        x2: np.ndarray,
        N: int,
    ) -> np.ndarray:
        """Look up C via per-parameter 2-D lazy splines, grouping
        samples by memoization key so each distinct parameter value
        triggers at most one spline build."""
        n = len(t1)
        result = np.empty((n, N))
        keys = [lazy._make_key(x1[k], x2[k]) for k in range(n)]
        seen: dict = {}
        for k, key in enumerate(keys):
            seen.setdefault(key, []).append(k)
        for key, idxs in seen.items():
            rep_k = idxs[0]
            splines = lazy.get_splines(x1[rep_k], x2[rep_k])
            sel = np.array(idxs)
            t1_sel = t1[sel]
            t2_sel = t2[sel]
            for a in range(N):
                result[sel, a] = splines[a](t1_sel, t2_sel, grid=False)
        return result


# ---------------------------------------------------------------------------
# Step 4: Contraction & Integration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DynamicCouplingPromise:
    """Defers coupling-tensor materialisation to QMC-sample time.

    When any entry in ``coupling_values`` is a callable (e.g. a
    spacetime-dependent non-local vertex such as demo2's
    ``κ^{(3)}(x₁,t₁; x₂,t₂; x₃,t₃)``), the usual pre-QMC
    :meth:`DiagramTerm.evaluate_coupling` path doesn't apply —
    the callable's output changes with each sample's QMC time
    coordinates.  This class packages everything needed for the
    per-sample path:

    - the list of static (``ndarray``) symbol values,
    - the list of dynamic (callable) symbol values,
    - for each dynamic symbol, the ψ-leg spatial-label tuple
      extracted from the :class:`DiagramTerm`'s ``coupling_sum`` so
      we can look up each leg's time and position per sample.

    The per-sample evaluator :meth:`evaluate_at` materialises the
    dynamic tensors using the sample's ``(times, positions)`` and
    then delegates to the static-path
    :meth:`DiagramTerm.evaluate_coupling` with a fully-numeric
    ``coupling_values`` dict — so the contraction code stays the
    same.
    """

    #: The parent :class:`DiagramTerm`.
    diagram_term: Any

    #: Static coupling values — already materialised arrays.
    static_values: dict

    #: Dynamic coupling values — ``{name: callable(n_list, t_list)
    #: -> ndarray}``.
    dynamic_values: dict

    #: Per-dynamic-symbol tuple of ψ-leg spatial labels, as they
    #: appear in the diagram's ``coupling_sum``.
    spatial_args_by_name: dict

    #: Component-index pins forwarded from ``build_integrand``.
    fixed_indices: dict

    def evaluate_at(
        self,
        times: dict,
        positions: dict,
    ) -> np.ndarray:
        """Materialise the dynamic callables at this sample's
        ``(times, positions)`` and return the per-sample
        ``coupling_array``.

        Args:
            times: ``{spatial_label: time_value}`` for all spatial
                labels referenced by any dynamic symbol's legs.
            positions: ``{spatial_label: position_value}`` same.
        """
        sample_cv = dict(self.static_values)
        for name, fn in self.dynamic_values.items():
            legs = self.spatial_args_by_name[name]
            t_list = np.array([times[s] for s in legs], dtype=float)
            n_list = np.array([positions[s] for s in legs], dtype=float)
            sample_cv[name] = np.asarray(fn(n_list, t_list))
        return self.diagram_term.evaluate_coupling(
            sample_cv, self.fixed_indices,
        )

    def evaluate_at_batch(
        self,
        label_t: dict,
        label_x: dict,
        n_samples: int,
    ) -> np.ndarray:
        """Vectorised batch evaluator -- returns a ``(n_samples,)``
        complex array of the contracted scalar coupling per sample.

        ``label_t`` and ``label_x`` map every spatial label appearing
        in any dynamic symbol's legs to a ``(n_samples,)`` time
        array / scalar-or-``(n_samples,)`` position respectively.

        Per-symbol fast path: when the user marks a callable
        ``vectorized=True`` (e.g. via
        :class:`~sft_wick.workflow.specs.NonLocalVertex(coupling_vectorized=True)`
        or by setting ``fn.vectorized = True``), the wrapped
        callable receives ``(m_legs, n_samples)`` arrays in a single
        call and returns a tensor of shape
        ``(n_samples,) + (N,)*order``. Otherwise we fall back to the
        ``n_samples`` per-sample calls of :meth:`evaluate_at`.

        Either way, we still loop in Python over ``n_samples`` to
        contract each sample's tensor down to a scalar via
        :meth:`DiagramTerm.evaluate_coupling`. Vectorising the
        contraction itself across a sample axis is the deferred
        prop-indexed-dynamic refactor and is not done here.

        Raises ``NotImplementedError`` if any sample's contracted
        coupling is not scalar (the prop-indexed case locked by
        ``DC1`` in ``tests/test_dynamic_coupling.py``).
        """
        # Pre-build (m, n_samples) leg arrays once per symbol, so the
        # per-sample loop only does scalar slicing.
        per_symbol_legs: dict = {}
        for name in self.dynamic_values:
            legs = self.spatial_args_by_name[name]
            t_2d = np.stack(
                [np.asarray(label_t[s], dtype=float) for s in legs],
                axis=0,
            )  # shape (m, n_samples)
            leg_x_arrs = []
            for s in legs:
                x_val = np.asarray(label_x[s], dtype=float)
                # The dynamic-coupling QMC path is currently
                # scalar-position-only. d-dim positions pass through
                # the static / non-dynamic path (homogeneity-aware
                # ``C_at_batch``) just fine -- only the user fn
                # contract here insists on scalar legs. If you need
                # vector positions plus a callable kappa^(m), wait
                # for the deferred dynamic-d-dim work or supply a
                # vectorized callable that pre-broadcasts the legs
                # itself.
                if x_val.ndim != 0:
                    raise NotImplementedError(
                        f"d-dim spatial positions not supported in "
                        f"the dynamic-coupling QMC path: symbol "
                        f"{name!r} leg at label {s!r} has position "
                        f"shape {x_val.shape}. Use scalar positions "
                        f"or a constant-tensor coupling for this "
                        f"vertex."
                    )
                leg_x_arrs.append(np.full((n_samples,), float(x_val)))
            n_arr = np.stack(leg_x_arrs, axis=0)  # shape (m, n_samples)
            per_symbol_legs[name] = (n_arr, t_2d)

        # Vectorised symbols: single fn call yields per-sample tensor
        # stack of shape (n_samples, *kappa_shape). For the rest, we
        # fall back to the per-sample call inside the loop.
        per_sample_tensors: dict = {}
        for name, fn in self.dynamic_values.items():
            if getattr(fn, "vectorized", False):
                n_arr, t_2d = per_symbol_legs[name]
                stacked = np.asarray(fn(n_arr, t_2d))
                if stacked.shape[0] != n_samples:
                    raise ValueError(
                        f"vectorized callable {name!r} returned shape "
                        f"{stacked.shape}; expected leading axis of "
                        f"length n_samples={n_samples}."
                    )
                per_sample_tensors[name] = stacked

        # Probe shape on sample 0 for an early NotImplementedError on
        # prop-indexed dynamic. Mirrors the per-sample loop in
        # ``integrate_moment_qmc_vectorized`` -- callers rely on this
        # raise to signal "use a static coupling tensor instead".
        sample_cv0 = dict(self.static_values)
        for name, fn in self.dynamic_values.items():
            if name in per_sample_tensors:
                sample_cv0[name] = per_sample_tensors[name][0]
            else:
                n_arr, t_2d = per_symbol_legs[name]
                sample_cv0[name] = np.asarray(fn(n_arr[:, 0], t_2d[:, 0]))
        coup0 = self.diagram_term.evaluate_coupling(
            sample_cv0, self.fixed_indices,
        )
        if coup0.ndim != 0:
            raise NotImplementedError(
                "Dynamic coupling with propagator-indexed "
                "contraction is not yet supported.  Either "
                "use a spacetime-constant coupling tensor, or "
                "provide a fully contracted scalar callable."
            )

        couplings = np.empty(n_samples, dtype=complex)
        couplings[0] = complex(coup0)
        for s in range(1, n_samples):
            sample_cv = dict(self.static_values)
            for name, fn in self.dynamic_values.items():
                if name in per_sample_tensors:
                    sample_cv[name] = per_sample_tensors[name][s]
                else:
                    n_arr, t_2d = per_symbol_legs[name]
                    sample_cv[name] = np.asarray(fn(n_arr[:, s], t_2d[:, s]))
            couplings[s] = complex(
                self.diagram_term.evaluate_coupling(
                    sample_cv, self.fixed_indices,
                )
            )
        return couplings


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
    #: When :attr:`dynamic_coupling` is set, this is a placeholder
    #: zero array and the real coupling value is computed per-sample
    #: in the integrator.
    coupling_array: np.ndarray

    #: Fixed component indices (e.g. ``{'a': 0, 'b': 1}``).
    #: Used by the QMC/GL integrators to resolve C-propagator
    #: component indices that are not summation variables.
    fixed_indices: dict[str, int] = field(default_factory=dict)

    #: Optional :class:`DynamicCouplingPromise` carrying per-sample
    #: coupling materialisation logic for spacetime-dependent
    #: (callable) coupling values.  ``None`` ⇒ the fully-static
    #: fast path is used (all coupling values were ndarrays).
    dynamic_coupling: Any = None

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
            # No propagator indices → scalar coupling, evaluate C without
            # summation, but still honour the integrand's
            # ``fixed_indices`` (observable component labels like
            # ``a``, ``b``) — without this the C matrix falls through
            # to ``C_mat.trace()`` and picks up a spurious factor of
            # N at order 0 for ``⟨φ_a(x) φ_b(y)⟩``-style observables.
            c_val = 1.0
            for sp_l, sp_r, il, ir in spatial.c_propagators:
                dir_l = spatial.direction_map[sp_l]
                dir_r = spatial.direction_map[sp_r]
                n_l = directions.get(dir_l, directions.get(sp_l))
                n_r = directions.get(dir_r, directions.get(sp_r))
                C_mat = cache.C_value(n_l, times[sp_l], n_r, times[sp_r])
                a = self._resolve_component(il, self.fixed_indices)
                b = self._resolve_component(ir, self.fixed_indices)
                if a is not None and b is not None:
                    c_val *= C_mat[a, b]
                else:
                    c_val *= C_mat.trace()
            return complex(r_val * complex(coeff) * c_val)

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
                # Literal index (e.g. '1') not in summation — resolve directly
                a = DiagramIntegrand._resolve_component(idx_name, {})
                if a is not None:
                    contracted = contracted * c_diag[a]
                else:
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

            # Merge the integrand's fixed component indices (e.g.
            # observable labels like ``a``, ``b``) into the
            # per-iteration summation ``idx_map``, matching the
            # vectorised path — without this merge, propagator legs
            # that reference observable labels fall through to
            # ``C_mat.trace()`` (spurious factor of N).
            idx_map = {
                **self.fixed_indices,
                **{name: val for name, val in zip(idx_names, pidx)},
            }

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

    @staticmethod
    def _resolve_group_x(
        spatial: "SpatialStructure",
        positions: dict[str, Any] | None,
        default: Any,
    ) -> dict[str, Any]:
        """Map each direction group (by its direction-variable name) to
        a single spatial coordinate.

        Priority for each group:
        1.  If the group contains an external point whose position the
            caller specified in ``positions``, use that.
        2.  Otherwise fall back to ``default`` (typically 0 for
            integration-only groups, or whatever ``direction`` kwarg
            supplies for backward compat).

        This is what flows into ``cache.C_at_batch`` as the endpoint
        x-value when the cache has a spatial table built (of any
        homogeneity kind).

        ``positions`` values may be scalars (1-D / legacy) or
        arbitrary-dimensional vectors. The ``default`` likewise may
        be a scalar or a vector. The returned dict preserves whatever
        shape the user passed -- downstream
        ``PropagatorCache.C_at_batch`` and the spatial Kappa2
        wrappers (``_SeparableTranslationKappa2``, ``_rotation_cos``,
        ...) all accept either form.
        """
        positions = positions or {}
        group_x: dict[str, Any] = {}
        for group in spatial.direction_groups:
            dvar_sample = spatial.direction_map[next(iter(group))]
            x_val: Any = default
            for p in group:
                if p in positions:
                    x_val = positions[p]
                    break
            group_x[dvar_sample] = x_val
        return group_x

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

    def integrate_moment_qmc(
        self,
        lambda_f: float,
        cache: PropagatorCache,
        t_min: float = 0.0,
        direction: Any = 0,
        n_samples: int = 2**14,
        seed: int | None = None,
        positions: dict[str, float] | None = None,
        integrate_over: Any = None,
    ) -> tuple[float, float]:
        """Integrate over all time variables using Quasi-Monte Carlo.

        Scalar-loop sibling of
        :meth:`integrate_moment_qmc_vectorized`; see that method's
        docstring for the ``integrate_over`` kwarg semantics.
        """
        from scipy.stats import qmc

        spatial = self.spatial
        int_vars_parents_first = list(reversed(spatial.time_integration_vars))
        ext_vars = list(spatial.external_points)

        ext_int_set = _resolve_integrate_over(integrate_over, ext_vars)
        ext_integrated = [v for v in ext_vars if v in ext_int_set]
        ext_fixed = [v for v in ext_vars if v not in ext_int_set]

        sobol_vars = ext_integrated + int_vars_parents_first
        n_ext_int = len(ext_integrated)
        n_total = len(sobol_vars)

        if _cache_has_spatial_table(cache):
            directions = self._resolve_group_x(spatial, positions, direction)
        else:
            dir_vars = set(spatial.direction_map.values())
            directions = {d: direction for d in dir_vars}

        if n_total == 0:
            fixed_times = {v: lambda_f for v in ext_fixed}
            val = self.evaluate(fixed_times, directions, cache)
            return (val.real, 0.0)

        parent_map: dict[str, list[str]] = defaultdict(list)
        for earlier, later in spatial.time_orderings:
            if earlier in int_vars_parents_first:
                parent_map[earlier].append(later)

        # Generate Sobol samples in [0,1]^d
        sampler = qmc.Sobol(d=n_total, seed=seed)
        u_samples = sampler.random(n_samples)  # (n_samples, n_total)

        # Evaluate integrand at each sample point
        values = np.empty(n_samples)
        span = lambda_f - t_min

        for s in range(n_samples):
            u = u_samples[s]
            times: dict[str, float] = {v: lambda_f for v in ext_fixed}
            jacobian = 1.0

            # Integrated externals: free in [t_min, lambda_f]
            for k in range(n_ext_int):
                t = t_min + u[k] * span
                times[ext_integrated[k]] = t
                jacobian *= span

            # Internal vars: bounded by causal parents
            for k, var in enumerate(int_vars_parents_first):
                idx = n_ext_int + k
                parents = parent_map.get(var, [])
                if parents:
                    hi = min(times[p] for p in parents if p in times)
                else:
                    hi = lambda_f
                lo = t_min
                width = hi - lo
                if width <= 0:
                    jacobian = 0.0
                    times[var] = lo
                else:
                    times[var] = lo + u[idx] * width
                    jacobian *= width

            if jacobian == 0.0:
                values[s] = 0.0
                continue

            result = self.evaluate(times, directions, cache)
            val = result.real if result.imag == 0 else abs(result)
            values[s] = val * jacobian

        estimate = float(np.mean(values))

        # Standard error from 8 batched sub-means
        n_batches = min(8, n_samples)
        batch_size = n_samples // n_batches
        batch_means = np.array([
            np.mean(values[i * batch_size:(i + 1) * batch_size])
            for i in range(n_batches)
        ])
        std_error = float(np.std(batch_means, ddof=1) / np.sqrt(n_batches))

        return (estimate, std_error)

    def integrate_moment_qmc_vectorized(
        self,
        lambda_f: float,
        cache: PropagatorCache,
        t_min: float = 0.0,
        direction: Any = 0,
        n_samples: int = 2**14,
        seed: int | None = None,
        positions: dict[str, float] | None = None,
        integrate_over: Any = None,
    ) -> tuple[float, float]:
        """Vectorized QMC integration — no Python loop over samples.

        Same algorithm as :meth:`integrate_moment_qmc` but evaluates
        all Sobol samples simultaneously using batch propagator lookups.
        Requires ``precompute_C_table`` for the batch C evaluation.

        For the ``iso_R + diag_C`` case with scalar coupling (no
        propagator indices), the integrand at each sample is::

            coupling × prod_R R(t_l, t_r) × prod_C C_diag(t_l, t_r)

        where all R and C values are looked up in batch.

        Args:
            positions: Optional mapping ``{point_name: spatial_coord}``
                (e.g. ``{'x': 0.0, 'y': 0.5}``).  When the cache has a
                spatial table built (any :attr:`~PropagatorCache.homogeneity`
                kind), the x-coordinate of each propagator endpoint is
                derived from its direction group's external-point
                position and flows through ``cache.C_at_batch``.
                Ignored when no spatial table is built (legacy
                behaviour — ``C_diagonal_batch`` has no x-dependence
                regardless).
            integrate_over: Controls which **external** points have
                their time integrated over ``[t_min, lambda_f]``.

                - ``None`` (default, physics observable):
                  **all externals held fixed at ``lambda_f``** —
                  this gives the equal-time correlator
                  ``⟨φ(t_f) φ(t_f)⟩`` that demo notebooks and MC
                  data compare against.
                - ``"all"``: all externals integrated — the
                  time-integrated moment
                  ``⟨∫₀^{t_f}φ(t) dt · ∫₀^{t_f}φ(t') dt'⟩``.
                  Useful e.g. for weak-lensing line-of-sight
                  integrals of the Sachs field.
                - Iterable of external point names (subset):
                  those listed are integrated, the rest fixed at
                  ``lambda_f``.  Enables mixed observables such as
                  an integrated source × a detector field.

                Internal vertex times are always integrated; their
                causal parent-map uses the (possibly fixed)
                external times as upper bounds.

        Returns:
            ``(estimate, std_error)``
        """
        from scipy.stats import qmc

        spatial = self.spatial
        int_vars_pf = list(reversed(spatial.time_integration_vars))
        ext_vars = list(spatial.external_points)

        # Partition externals into "integrated" vs "fixed at lambda_f".
        ext_int_set = _resolve_integrate_over(integrate_over, ext_vars)
        ext_integrated = [v for v in ext_vars if v in ext_int_set]
        ext_fixed = [v for v in ext_vars if v not in ext_int_set]
        n_ext_int = len(ext_integrated)

        # Sobol-dimensioned vars = integrated externals + internals.
        sobol_vars = ext_integrated + int_vars_pf
        n_total = len(sobol_vars)

        if n_total == 0:
            # No integration — purely evaluate at the fixed external
            # times.  Use ``_resolve_group_x`` to thread user-supplied
            # ``positions`` into the ``directions`` dict so the C
            # propagator still picks up the spatial factor (e.g. the
            # ``exp(-r/σ_x)`` at order 0 for observables like
            # ``⟨φ(x) φ(y)⟩``).
            if _cache_has_spatial_table(cache):
                directions = self._resolve_group_x(
                    spatial, positions, direction,
                )
            else:
                dir_vars = set(spatial.direction_map.values())
                directions = {d: direction for d in dir_vars}
            fixed_times = {v: lambda_f for v in ext_fixed}
            val = self.evaluate(fixed_times, directions, cache)
            return (val.real, 0.0)

        # Build causal parent map (per-internal-var upper bound list).
        parent_map: dict[str, list[str]] = defaultdict(list)
        for earlier, later in spatial.time_orderings:
            if earlier in int_vars_pf:
                parent_map[earlier].append(later)

        # Sobol samples
        sampler = qmc.Sobol(d=n_total, seed=seed)
        u = sampler.random(n_samples)
        span = lambda_f - t_min

        # Map u -> times with causal bounds (vectorized)
        times_arr = np.empty((n_samples, n_total))
        jacobians = np.ones(n_samples)

        # Integrated external vars: free in [t_min, lambda_f]
        for k in range(n_ext_int):
            times_arr[:, k] = t_min + u[:, k] * span
            jacobians *= span

        # Internal vars: bounded by parents.  Parents may be
        # (a) integrated externals — pull from times_arr; (b) fixed
        # externals — use the fixed ``lambda_f`` value; (c) other
        # internal vars — pull from times_arr.
        for k, var in enumerate(int_vars_pf):
            idx = n_ext_int + k
            parents = parent_map.get(var, [])
            if parents:
                hi = np.full(n_samples, lambda_f)
                for p in parents:
                    if p in int_vars_pf:
                        p_idx = n_ext_int + int_vars_pf.index(p)
                        hi = np.minimum(hi, times_arr[:, p_idx])
                    elif p in ext_integrated:
                        p_idx = ext_integrated.index(p)
                        hi = np.minimum(hi, times_arr[:, p_idx])
                    else:
                        # Fixed external — its time is lambda_f.
                        hi = np.minimum(hi, lambda_f)
            else:
                hi = np.full(n_samples, lambda_f)

            width = hi - t_min
            valid = width > 0
            times_arr[:, idx] = np.where(valid, t_min + u[:, idx] * width, t_min)
            jacobians = np.where(valid, jacobians * width, 0.0)

        # Build variable-name to column lookup.  Fixed externals are
        # NOT in times_arr; they get a special lookup that returns a
        # constant ``lambda_f`` array.
        var_to_col = {var: i for i, var in enumerate(sobol_vars)}
        fixed_t = np.full(n_samples, lambda_f)

        def _times(var: str) -> np.ndarray:
            """Return the per-sample time array for ``var`` — pulls
            from ``times_arr`` when integrated, or the constant
            ``lambda_f`` array when fixed."""
            col = var_to_col.get(var)
            if col is not None:
                return times_arr[:, col]
            return fixed_t

        # --- Vectorized integrand evaluation ---
        dt = self.diagram_term
        coeff = self.coupling_array
        prop_idx = dt.propagator_indices

        # R product (vectorized)
        r_product = np.ones(n_samples)
        for sl, sr in spatial.r_propagators:
            t_l = _times(sl)
            t_r = _times(sr)
            r_product *= cache.R_time_batch(t_l, t_r)

        fi = self.fixed_indices

        # --- Spatial-aware dispatch ---
        # Decide once per call how to evaluate each C propagator.
        # We route through ``C_at_batch`` whenever the cache has a
        # spatial table built (of any homogeneity kind); otherwise
        # fall back to the legacy ``C_diagonal_batch`` which ignores
        # x and matches the pre-Phase-5 behaviour bit-identically.
        spatial_aware = _cache_has_spatial_table(cache)
        group_x: dict[str, float] | None = None
        if spatial_aware:
            group_x = self._resolve_group_x(spatial, positions, direction)

        def _lookup_C(sp_l: str, sp_r: str, t_l: np.ndarray,
                      t_r: np.ndarray) -> np.ndarray:
            """Evaluate C at the batch of (t_l, t_r) samples, carrying
            the appropriate x-coordinate when the cache is
            spatial-aware."""
            if not spatial_aware:
                return cache.C_diagonal_batch(t_l, t_r)
            x_l = group_x[spatial.direction_map[sp_l]]  # type: ignore[index]
            x_r = group_x[spatial.direction_map[sp_r]]  # type: ignore[index]
            return cache.C_at_batch(t_l, t_r, x_l, x_r)

        if self.dynamic_coupling is not None:
            # --- Per-sample (dynamic) coupling path ---
            # Triggered when any coupling_value passed to
            # ``build_integrand`` was callable (e.g. a
            # spacetime-dependent non-local vertex like demo2's
            # ``κ^{(3)}``).  Per-sample cost is one callable
            # invocation per dynamic symbol × one cheap
            # :meth:`DiagramTerm.evaluate_coupling` substitution.
            #
            # Vectorisation strategy:
            # * ``r_product`` is already (n_samples,)-batched above.
            # * ``c_product`` is hoisted out of the per-sample loop
            #   and computed via batched ``C_at_batch`` calls in
            #   parity with the static scalar path below (line ~2235).
            # * Per-symbol ``label_t`` / ``label_x`` arrays are built
            #   ONCE so the inner loop only does scalar slicing.
            # * If all active dynamic symbols set
            #   ``vectorized=True``, ``DynamicCouplingPromise``
            #   collects (n_samples,) coupling values in a single
            #   call instead of n_samples calls; otherwise we fall
            #   through to the per-sample loop.
            all_spatial_labels = set(spatial.direction_map.keys())
            _promise = self.dynamic_coupling

            # --- A: hoist C-propagator lookup out of per-sample loop. ---
            c_product = np.ones(n_samples)
            for sp_l, sp_r, il, ir in spatial.c_propagators:
                t_l = _times(sp_l)
                t_r = _times(sp_r)
                C_batch = _lookup_C(sp_l, sp_r, t_l, t_r)
                a_i = (
                    DiagramIntegrand._resolve_component(il, fi)
                    if il is not None else None
                )
                b_i = (
                    DiagramIntegrand._resolve_component(ir, fi)
                    if ir is not None else None
                )
                if a_i is not None and b_i is not None:
                    c_product *= C_batch[:, a_i]
                else:
                    c_product *= C_batch.sum(axis=1)

            # --- B: pre-build per-symbol-leg time / position arrays. ---
            label_t = {lab: _times(lab) for lab in all_spatial_labels}
            if spatial_aware:
                # group_x is computed above (line ~2146) when the
                # cache is spatial-aware; reuse it.
                label_x = {
                    lab: group_x[spatial.direction_map[lab]]  # type: ignore[index]
                    for lab in all_spatial_labels
                }
            else:
                label_x = {
                    lab: float(direction)
                    for lab in all_spatial_labels
                }

            couplings = _promise.evaluate_at_batch(
                label_t=label_t,
                label_x=label_x,
                n_samples=n_samples,
            )
            # ``couplings`` is a (n_samples,) complex/float array of
            # the contracted scalar coupling at each sample. The
            # NotImplementedError for prop-indexed dynamic is raised
            # inside ``evaluate_at_batch``.
            values = (
                couplings.real * r_product * c_product * jacobians
            )

        elif not prop_idx:
            # Scalar coupling path (iso_R + iso_C or no prop indices)
            c_product = np.ones(n_samples)
            for sp_l, sp_r, il, ir in spatial.c_propagators:
                t_l = _times(sp_l)
                t_r = _times(sp_r)
                C_diag_batch = _lookup_C(sp_l, sp_r, t_l, t_r)
                if il is not None and ir is not None:
                    a = DiagramIntegrand._resolve_component(il, fi)
                    b = DiagramIntegrand._resolve_component(ir, fi)
                    if a is not None and b is not None:
                        c_product *= C_diag_batch[:, a]
                    else:
                        c_product *= C_diag_batch.sum(axis=1)
                else:
                    c_product *= C_diag_batch.sum(axis=1)

            values = r_product * complex(coeff).real * c_product * jacobians

        else:
            # Propagator-indexed coupling: loop over index combinations
            idx_names = [name for name, _ in prop_idx]
            prop_shape = tuple(dim for _, dim in prop_idx)
            values = np.zeros(n_samples)

            for pidx in np.ndindex(*prop_shape):
                c_val = complex(coeff[pidx]).real if coeff.ndim > 0 else complex(coeff).real
                if abs(c_val) < 1e-20:
                    continue

                idx_map = {**fi, **dict(zip(idx_names, pidx))}
                c_prod = np.ones(n_samples)
                for sp_l, sp_r, il, ir in spatial.c_propagators:
                    t_l = _times(sp_l)
                    t_r = _times(sp_r)
                    C_diag_batch = _lookup_C(sp_l, sp_r, t_l, t_r)
                    a = DiagramIntegrand._resolve_component(il, idx_map)
                    b = DiagramIntegrand._resolve_component(ir, idx_map)
                    if a is not None and b is not None:
                        c_prod *= C_diag_batch[:, a]
                    else:
                        c_prod *= C_diag_batch.sum(axis=1)

                values += c_val * r_product * c_prod * jacobians

        # Mask invalid samples
        values = np.where(jacobians > 0, values, 0.0)
        estimate = float(np.mean(values))

        # Error estimate
        n_batches = min(8, n_samples)
        batch_size = n_samples // n_batches
        batch_means = np.array([
            np.mean(values[i * batch_size:(i + 1) * batch_size])
            for i in range(n_batches)
        ])
        std_error = float(np.std(batch_means, ddof=1) / np.sqrt(n_batches))

        return (estimate, std_error)

    def integrate_moment_gauss_legendre(
        self,
        lambda_f: float,
        cache: PropagatorCache,
        t_min: float = 0.0,
        direction: Any = 0,
        n_gauss: int = 8,
        positions: dict[str, float] | None = None,
        integrate_over: Any = None,
    ) -> tuple[float, float]:
        """Tensor-product Gauss-Legendre quadrature on the causal
        simplex.

        Mirrors :meth:`integrate_moment_qmc_vectorized` node-for-node
        -- the SAME causal-simplex mapping (parents → upper bounds →
        Jacobians) and the SAME vectorised batch path through
        :meth:`PropagatorCache.R_time_batch` and
        :meth:`PropagatorCache.C_at_batch` -- but replaces the Sobol
        ``n_samples`` quasi-random nodes with the deterministic
        ``n_gauss^d`` tensor product of 1-D Gauss-Legendre nodes
        (mapped from ``[-1, 1]`` to ``[0, 1]``).

        For smooth integrands (a finite product of exponentials --
        the typical R/C/κ kernel structure) this gives **exponential
        convergence in n_gauss**, vastly outperforming Sobol QMC at
        modest dimensionality.  In particular, demo2's FK channel
        (4D smooth integrand on a causal simplex of area
        ``t_f^2 / 2``) is dominated at large ``t_f`` by a narrow
        peak band of area ``~ σ_t/γ`` near the upper-right corner;
        Sobol QMC under-resolves it unless ``n_samples`` is enormous,
        while ``n_gauss=8`` (4096 nodes for d=4) recovers the
        notebook's hand-derived value to ~5 sig figs.

        **Cost trade-off.**  Tensor-product GL scales as
        ``n_gauss^d`` -- fine for ``d ≤ 5`` (demo2 FK), fast at
        ``d ≤ 4``, painful at ``d ≥ 7``.  For high-d integrands
        prefer ``method='qmc_vectorized'`` with a large ``n_samples``
        instead.

        Args:
            n_gauss: Number of GL nodes per dimension.  ``n=8`` is
                a good default (handles up to degree-15 polynomial
                exactly).
            positions, integrate_over: Same as
                :meth:`integrate_moment_qmc_vectorized`.

        Returns:
            ``(estimate, 0.0)`` -- GL is deterministic so there is
            no statistical error to report.  The 0.0 mirrors the
            return shape of the QMC variants for downstream code.
        """
        from numpy.polynomial.legendre import leggauss

        spatial = self.spatial
        int_vars_pf = list(reversed(spatial.time_integration_vars))
        ext_vars = list(spatial.external_points)

        ext_int_set = _resolve_integrate_over(integrate_over, ext_vars)
        ext_integrated = [v for v in ext_vars if v in ext_int_set]
        ext_fixed = [v for v in ext_vars if v not in ext_int_set]
        n_ext_int = len(ext_integrated)

        gl_vars = ext_integrated + int_vars_pf
        n_total = len(gl_vars)

        if n_total == 0:
            # No integration -- defer to the standard non-integrated
            # evaluation path (parity with QMC vectorised).
            if _cache_has_spatial_table(cache):
                directions = self._resolve_group_x(
                    spatial, positions, direction,
                )
            else:
                dir_vars = set(spatial.direction_map.values())
                directions = {d: direction for d in dir_vars}
            fixed_times = {v: lambda_f for v in ext_fixed}
            val = self.evaluate(fixed_times, directions, cache)
            return (val.real, 0.0)

        # 1-D Gauss-Legendre nodes / weights mapped from [-1, 1] to [0, 1].
        nodes_1d, weights_1d = leggauss(n_gauss)
        u_1d = (nodes_1d + 1) / 2
        w_1d = weights_1d / 2

        # Tensor product of n_total dims.  ``np.meshgrid(..., indexing='ij')``
        # gives arrays of shape (n_gauss,) * n_total; ravel and stack to
        # (n_pts, n_total).
        mesh = np.meshgrid(*([u_1d] * n_total), indexing="ij")
        u = np.stack([m.ravel() for m in mesh], axis=-1)
        w_mesh = np.meshgrid(*([w_1d] * n_total), indexing="ij")
        node_weights = np.prod(
            np.stack([wm.ravel() for wm in w_mesh], axis=-1), axis=1
        )
        n_samples = u.shape[0]  # = n_gauss ** n_total

        # --- Causal mapping: identical to QMC vectorised path. ---
        parent_map: dict[str, list[str]] = defaultdict(list)
        for earlier, later in spatial.time_orderings:
            if earlier in int_vars_pf:
                parent_map[earlier].append(later)

        span = lambda_f - t_min
        times_arr = np.empty((n_samples, n_total))
        jacobians = np.ones(n_samples)

        for k in range(n_ext_int):
            times_arr[:, k] = t_min + u[:, k] * span
            jacobians *= span

        for k, var in enumerate(int_vars_pf):
            idx = n_ext_int + k
            parents = parent_map.get(var, [])
            if parents:
                hi = np.full(n_samples, lambda_f)
                for p in parents:
                    if p in int_vars_pf:
                        p_idx = n_ext_int + int_vars_pf.index(p)
                        hi = np.minimum(hi, times_arr[:, p_idx])
                    elif p in ext_integrated:
                        p_idx = ext_integrated.index(p)
                        hi = np.minimum(hi, times_arr[:, p_idx])
                    else:
                        hi = np.minimum(hi, lambda_f)
            else:
                hi = np.full(n_samples, lambda_f)

            width = hi - t_min
            valid = width > 0
            times_arr[:, idx] = np.where(valid, t_min + u[:, idx] * width, t_min)
            jacobians = np.where(valid, jacobians * width, 0.0)

        var_to_col = {var: i for i, var in enumerate(gl_vars)}
        fixed_t = np.full(n_samples, lambda_f)

        def _times(var: str) -> np.ndarray:
            col = var_to_col.get(var)
            if col is not None:
                return times_arr[:, col]
            return fixed_t

        # --- Vectorised integrand evaluation: identical to QMC path. ---
        dt = self.diagram_term
        coeff = self.coupling_array
        prop_idx = dt.propagator_indices

        r_product = np.ones(n_samples)
        for sl, sr in spatial.r_propagators:
            r_product *= cache.R_time_batch(_times(sl), _times(sr))

        fi = self.fixed_indices
        spatial_aware = _cache_has_spatial_table(cache)
        group_x: dict[str, float] | None = None
        if spatial_aware:
            group_x = self._resolve_group_x(spatial, positions, direction)

        def _lookup_C(sp_l, sp_r, t_l, t_r):
            if not spatial_aware:
                return cache.C_diagonal_batch(t_l, t_r)
            x_l = group_x[spatial.direction_map[sp_l]]  # type: ignore[index]
            x_r = group_x[spatial.direction_map[sp_r]]  # type: ignore[index]
            return cache.C_at_batch(t_l, t_r, x_l, x_r)

        if self.dynamic_coupling is not None:
            all_spatial_labels = set(spatial.direction_map.keys())
            _promise = self.dynamic_coupling

            c_product = np.ones(n_samples)
            for sp_l, sp_r, il, ir in spatial.c_propagators:
                t_l = _times(sp_l)
                t_r = _times(sp_r)
                C_batch = _lookup_C(sp_l, sp_r, t_l, t_r)
                a_i = (
                    DiagramIntegrand._resolve_component(il, fi)
                    if il is not None else None
                )
                b_i = (
                    DiagramIntegrand._resolve_component(ir, fi)
                    if ir is not None else None
                )
                if a_i is not None and b_i is not None:
                    c_product *= C_batch[:, a_i]
                else:
                    c_product *= C_batch.sum(axis=1)

            label_t = {lab: _times(lab) for lab in all_spatial_labels}
            if spatial_aware:
                label_x = {
                    lab: group_x[spatial.direction_map[lab]]  # type: ignore[index]
                    for lab in all_spatial_labels
                }
            else:
                label_x = {
                    lab: float(direction)
                    for lab in all_spatial_labels
                }

            couplings = _promise.evaluate_at_batch(
                label_t=label_t,
                label_x=label_x,
                n_samples=n_samples,
            )
            values = couplings.real * r_product * c_product * jacobians

        elif not prop_idx:
            c_product = np.ones(n_samples)
            for sp_l, sp_r, il, ir in spatial.c_propagators:
                t_l = _times(sp_l)
                t_r = _times(sp_r)
                C_diag_batch = _lookup_C(sp_l, sp_r, t_l, t_r)
                if il is not None and ir is not None:
                    a = DiagramIntegrand._resolve_component(il, fi)
                    b = DiagramIntegrand._resolve_component(ir, fi)
                    if a is not None and b is not None:
                        c_product *= C_diag_batch[:, a]
                    else:
                        c_product *= C_diag_batch.sum(axis=1)
                else:
                    c_product *= C_diag_batch.sum(axis=1)
            values = r_product * complex(coeff).real * c_product * jacobians

        else:
            idx_names = [name for name, _ in prop_idx]
            prop_shape = tuple(dim for _, dim in prop_idx)
            values = np.zeros(n_samples)
            for pidx in np.ndindex(*prop_shape):
                c_val = (
                    complex(coeff[pidx]).real if coeff.ndim > 0
                    else complex(coeff).real
                )
                if abs(c_val) < 1e-20:
                    continue
                idx_map = {**fi, **dict(zip(idx_names, pidx))}
                c_prod = np.ones(n_samples)
                for sp_l, sp_r, il, ir in spatial.c_propagators:
                    t_l = _times(sp_l)
                    t_r = _times(sp_r)
                    C_diag_batch = _lookup_C(sp_l, sp_r, t_l, t_r)
                    a = DiagramIntegrand._resolve_component(il, idx_map)
                    b = DiagramIntegrand._resolve_component(ir, idx_map)
                    if a is not None and b is not None:
                        c_prod *= C_diag_batch[:, a]
                    else:
                        c_prod *= C_diag_batch.sum(axis=1)
                values += c_val * r_product * c_prod * jacobians

        # --- GL aggregation: weighted sum (NOT mean). ---
        # Each tensor-product node carries weight ``w_i = prod_d w_1d[i_d]``.
        # The integrand on the unit cube is therefore
        #   integral = sum_i (values[i] * w_i)
        # where ``values[i]`` already includes the causal-simplex
        # Jacobian.  Compare to QMC which uses ``mean(values) =
        # sum(values) / n_samples`` (Monte Carlo estimator).
        values = np.where(jacobians > 0, values, 0.0)
        estimate = float(np.sum(values * node_weights))

        return (estimate, 0.0)

    def integrate_moment_nquad(
        self,
        lambda_f: float,
        cache: PropagatorCache,
        t_min: float = 0.0,
        direction: Any = 0,
        positions: dict[str, float] | None = None,
        integrate_over: Any = None,
    ) -> tuple[float, float]:
        """Integrate over time variables using nested adaptive
        quadrature.

        See :meth:`integrate_moment_qmc_vectorized` for the
        ``integrate_over`` kwarg semantics (external-time partition
        into integrated vs fixed-at-``lambda_f``).
        """
        from scipy.integrate import nquad as _nquad

        spatial = self.spatial
        int_vars = list(spatial.time_integration_vars)
        ext_vars = list(spatial.external_points)

        ext_int_set = _resolve_integrate_over(integrate_over, ext_vars)
        ext_integrated = [v for v in ext_vars if v in ext_int_set]
        ext_fixed = [v for v in ext_vars if v not in ext_int_set]

        # Quadrature variables = internals + integrated externals.
        all_vars = int_vars + ext_integrated
        n_int = len(int_vars)
        n_total = len(all_vars)

        if _cache_has_spatial_table(cache):
            directions = self._resolve_group_x(spatial, positions, direction)
        else:
            dir_vars = set(spatial.direction_map.values())
            directions = {d: direction for d in dir_vars}

        fixed_times = {v: lambda_f for v in ext_fixed}

        if n_total == 0:
            val = self.evaluate(fixed_times, directions, cache)
            return (val.real if val.imag == 0 else abs(val), 0.0)

        # Spacetime-dependent (callable) couplings are NOT supported
        # by the nquad path: ``self.evaluate`` uses the static
        # ``coupling_array`` which is a placeholder zero array when
        # the integrand was built from a callable κ.  Multiplying by
        # 0 would silently return 0 for every diagram with a
        # dynamic vertex (a latent bug pre-2026-04).  We refuse
        # explicitly and point to ``method='gauss_legendre'`` -- a
        # tensor-product GL rule with deterministic exponential
        # convergence on smooth integrands, which is the natural
        # match for diagrams that have callable couplings (and
        # vastly outperforms 4D adaptive nquad in practice anyway).
        if self.dynamic_coupling is not None:
            raise NotImplementedError(
                "method='nquad' does not support spacetime-dependent "
                "(callable) couplings.  Use method='gauss_legendre' "
                "(deterministic, exponential convergence on smooth "
                "integrands, matches the notebook hand-derivation) "
                "or method='qmc_vectorized' (Sobol QMC, recommended "
                "for high-d diagrams) instead."
            )

        def f(*args: float) -> float:
            times = dict(fixed_times)
            for i, var in enumerate(all_vars):
                times[var] = args[i]
            result = self.evaluate(times, directions, cache)
            return result.real if result.imag == 0 else abs(result)

        # Causal bounds for internal vars
        upper_bounds: dict[str, list[str]] = defaultdict(list)
        for earlier, later in spatial.time_orderings:
            if earlier in int_vars:
                upper_bounds[earlier].append(later)

        ranges: list = []
        for i, var in enumerate(all_vars):
            if i >= n_int:
                ranges.append((t_min, lambda_f))
            else:
                ub_sources = upper_bounds.get(var, [])
                if not ub_sources:
                    ranges.append((t_min, lambda_f))
                else:
                    # Fixed externals contribute a constant upper bound
                    # ``lambda_f`` (no longer showing up in later_args).
                    ub_dyn = [src for src in ub_sources
                              if src not in fixed_times]

                    def make_bound(
                        ub: list[str], avars: list[str],
                        lo: float, hi: float, cur_i: int,
                    ) -> Callable:
                        def bound_func(*later_args: float) -> tuple[float, float]:
                            hi_vals = [hi]
                            for src in ub:
                                j = avars.index(src) - cur_i - 1
                                if 0 <= j < len(later_args):
                                    hi_vals.append(later_args[j])
                                else:
                                    hi_vals.append(hi)
                            return (lo, min(hi_vals))
                        return bound_func
                    ranges.append(
                        make_bound(ub_dyn, all_vars, t_min, lambda_f, i)
                    )

        val, err = _nquad(f, ranges)
        return (val, err)


def _cache_supports_batch_c(cache: "PropagatorCache") -> bool:
    """Whether ``cache`` can evaluate C propagators in batch.

    True when the cache either has a pre-computed spline table
    (``_c_splines is not None`` on :class:`PropagatorCache`) or is a
    custom cache with a native batch implementation that sets
    ``_c_splines`` to a truthy sentinel (e.g. the analytical cache
    used in ``examples/demo1`` and ``tests/test_deductive_numerics``).
    """
    return getattr(cache, "_c_splines", None) is not None


def _cache_has_spatial_table(cache: "PropagatorCache") -> bool:
    """Whether ``cache`` has been equipped with any spatial
    (x-aware) table — full or lazy, any homogeneity kind.

    Used by the integrators to decide whether to route each C
    propagator through the spatial-aware ``C_at_batch`` path or the
    legacy ``C_diagonal_batch`` path (which ignores x).  False when
    the cache was built with only the legacy ``precompute_C_table``
    or when it is a custom cache that doesn't implement the new
    spatial attributes — in both cases the legacy path is
    bit-identical to pre-Phase-5 behaviour.
    """
    if getattr(cache, "_closed_form_only", False):
        # Closed-form-only mode: no spline at all, but the
        # ``C_at_batch`` override IS spatial-aware (it routes the
        # per-sample positions straight into the user's c_fn).
        return True
    names = (
        "_c_translation_splines",
        "_lazy_translation",
        "_c_rotation_splines",
        "_lazy_rotation",
        "_c_general_interpolators",
        "_lazy_general",
    )
    return any(getattr(cache, n, None) is not None for n in names)


def integrate_moment(
    integrand: DiagramIntegrand,
    lambda_f: float,
    cache: PropagatorCache,
    t_min: float = 0.0,
    direction: Any = 0,
    method: str = "qmc",
    n_samples: int = 2**14,
    seed: int | None = None,
    positions: dict[str, float] | None = None,
    integrate_over: Any = None,
    n_gauss: int = 8,
) -> tuple[float, float]:
    """Integrate a diagram's contribution over all time variables.

    Convenience wrapper around
    :meth:`DiagramIntegrand.integrate_moment_qmc`,
    :meth:`DiagramIntegrand.integrate_moment_qmc_vectorized`, and
    :meth:`DiagramIntegrand.integrate_moment_nquad`.

    **Dispatch logic.**  With ``method='qmc'`` (default) the function
    auto-selects the fastest QMC path compatible with the supplied
    cache:

    - If the cache supports batched C evaluation (either
      ``PropagatorCache.precompute_C_table`` has been called, or a
      custom cache implements ``C_diagonal_batch`` natively) →
      :meth:`integrate_moment_qmc_vectorized` (~200× faster than
      the scalar loop on typical workloads).
    - Otherwise → :meth:`integrate_moment_qmc` (scalar Python loop).

    Users who explicitly want the scalar loop (e.g. for debugging
    or with a non-batch-capable custom cache) can pass
    ``method='qmc_scalar'``.

    For best performance, call
    :meth:`PropagatorCache.precompute_C_table` before integrating so
    the vectorised path is selected automatically.

    Args:
        integrand: A :class:`DiagramIntegrand` built from a
            :class:`~sft_wick.perturbation.DiagramTerm`.
        lambda_f: Upper integration limit for external times.
        cache: A :class:`PropagatorCache` (or compatible custom
            cache).
        t_min: Lower time bound (default 0).
        direction: Direction value for all spatial points.
        method: ``'qmc'`` (default, auto-selects vectorised vs
            scalar), ``'qmc_scalar'`` (force scalar loop),
            ``'qmc_vectorized'`` (force vectorised; raises if cache
            doesn't support batch), or ``'nquad'`` (nested adaptive
            quadrature).
        n_samples: Number of Sobol samples (QMC only, should be
            a power of 2).
        seed: Random seed for reproducibility (QMC only).

    Returns:
        ``(estimate, error)`` tuple.
    """
    if method == "qmc":
        if _cache_supports_batch_c(cache):
            return integrand.integrate_moment_qmc_vectorized(
                lambda_f, cache, t_min=t_min, direction=direction,
                n_samples=n_samples, seed=seed, positions=positions,
                integrate_over=integrate_over,
            )
        return integrand.integrate_moment_qmc(
            lambda_f, cache, t_min=t_min, direction=direction,
            n_samples=n_samples, seed=seed, positions=positions,
            integrate_over=integrate_over,
        )
    elif method == "qmc_vectorized":
        return integrand.integrate_moment_qmc_vectorized(
            lambda_f, cache, t_min=t_min, direction=direction,
            n_samples=n_samples, seed=seed, positions=positions,
            integrate_over=integrate_over,
        )
    elif method == "qmc_scalar":
        return integrand.integrate_moment_qmc(
            lambda_f, cache, t_min=t_min, direction=direction,
            n_samples=n_samples, seed=seed, positions=positions,
            integrate_over=integrate_over,
        )
    elif method == "nquad":
        return integrand.integrate_moment_nquad(
            lambda_f, cache, t_min=t_min, direction=direction,
            positions=positions,
            integrate_over=integrate_over,
        )
    elif method == "gauss_legendre":
        return integrand.integrate_moment_gauss_legendre(
            lambda_f, cache, t_min=t_min, direction=direction,
            n_gauss=n_gauss, positions=positions,
            integrate_over=integrate_over,
        )
    else:
        raise ValueError(
            f"Unknown method {method!r}; use 'qmc', 'qmc_scalar', "
            f"'qmc_vectorized', 'nquad', or 'gauss_legendre'"
        )


def _eval_single_diagram(
    dt, coupling_values, fixed_indices, lambda_f, cache, t_min, direction,
    method, n_samples, seed, positions, integrate_over, n_gauss,
):
    """Evaluate one diagram term end-to-end (build integrand + integrate).

    Top-level function so it is picklable for multiprocessing.
    """
    ig = dt.build_integrand(coupling_values, fixed_indices)
    return integrate_moment(
        ig, lambda_f, cache,
        t_min=t_min, direction=direction,
        method=method, n_samples=n_samples, seed=seed,
        positions=positions, integrate_over=integrate_over,
        n_gauss=n_gauss,
    )


def integrate_diagrams(
    diagram_terms: list,
    coupling_values: dict,
    lambda_f: float,
    cache: "PropagatorCache",
    t_min: float = 0.0,
    direction: Any = 0,
    method: str = "qmc",
    n_samples: int = 2**14,
    seed: int | None = None,
    fixed_indices: dict[str, int] | None = None,
    n_jobs: int = 1,
    positions: dict[str, Any] | None = None,
    integrate_over: Any = None,
    n_gauss: int = 8,
) -> tuple[float, list[tuple[float, float]]]:
    """Integrate a batch of diagram terms, optionally in parallel.

    Builds integrands and evaluates all diagrams, returning the total
    and per-diagram results.  When ``n_jobs != 1``, diagrams are
    evaluated in parallel using :mod:`joblib`.

    .. note::
        ``n_jobs`` defaults to ``1`` (sequential), matching the L1
        ``Expansion.evaluate`` / ``Expansion.sweep`` defaults. Pass
        ``-1`` to use all CPU cores. Sequential and parallel paths
        are bit-identical when the QMC seed is fixed.

    Args:
        diagram_terms: List of :class:`~sft_wick.perturbation.DiagramTerm`.
        coupling_values: Coupling tensor arrays (e.g. ``{'F': F_code}``).
        lambda_f: Upper integration limit.
        cache: A :class:`PropagatorCache`.
        t_min: Lower time bound (default 0).
        direction: Direction value for all spatial points.
        method: ``'qmc'`` or ``'nquad'``.
        n_samples: Number of Sobol samples (QMC only).
        seed: Random seed (QMC only).
        fixed_indices: Optional pinned index values for
            :meth:`~sft_wick.perturbation.DiagramTerm.build_integrand`.
        n_jobs: Number of parallel jobs.  ``1`` = sequential
            (default, safe under nested-callers), ``-1`` = all CPU
            cores.  A built-in guard falls
            back to sequential for ``len(diagram_terms) <= 2`` to
            avoid joblib's ~1 s startup overhead on trivial batches.
            Requires :mod:`joblib` when ``n_jobs != 1``.

    Returns:
        ``(total, details)`` where *total* is the scalar sum and
        *details* is a list of ``(estimate, error)`` per diagram.
    """
    if not diagram_terms:
        return (0.0, [])

    if n_jobs == 1 or len(diagram_terms) <= 2:
        # Sequential — no overhead
        details = []
        for dt in diagram_terms:
            ig = dt.build_integrand(coupling_values, fixed_indices)
            val, err = integrate_moment(
                ig, lambda_f, cache,
                t_min=t_min, direction=direction,
                method=method, n_samples=n_samples, seed=seed,
                positions=positions, integrate_over=integrate_over,
                n_gauss=n_gauss,
            )
            details.append((val, err))
    else:
        # Parallel via joblib
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_eval_single_diagram)(
                dt, coupling_values, fixed_indices,
                lambda_f, cache, t_min, direction,
                method, n_samples, seed, positions, integrate_over,
                n_gauss,
            )
            for dt in diagram_terms
        )
        details = list(results)

    total = sum(v for v, _ in details)
    return (total, details)


# ---------------------------------------------------------------------------
# Two-point correlation function integration
# ---------------------------------------------------------------------------


def integrate_two_point_qmc(
    integrands: list["DiagramIntegrand"],
    t_f: float,
    positions: dict[str, float],
    cache: "PropagatorCache",
    t_min: float = 0.0,
    n_samples: int = 2**14,
    seed: int | None = None,
) -> tuple[float, float]:
    """Vectorised QMC integration for two-point correlation functions.

    Evaluates the sum of diagram integrands at external time *t_f*
    with spatial points at the positions given by *positions*.  This
    is the standard entry point for computing correlators such as
    ``⟨φ(x, t) φ(y, t)⟩`` where *x* and *y* may differ.

    Unlike :func:`integrate_moment`, which assigns a single direction
    value to all spatial points, this function supports per-point
    spatial positions and applies the appropriate spatial factor to
    each correlation propagator automatically.

    The spatial factor for a *C*-propagator connecting points at
    positions *n₁* and *n₂* is computed as::

        κ²(n₁, t_ref; n₂, t_ref) / κ²(0, t_ref; 0, t_ref)

    which is exact for separable kernels (where the factor is
    time-independent) and a good approximation otherwise.

    Requires :meth:`PropagatorCache.precompute_C_table` to have been
    called for the batch *C* evaluation fast path.

    Args:
        integrands: List of :class:`DiagramIntegrand` objects (one per
            non-vanishing Feynman diagram).
        t_f: External (observation) time — all external points are set
            to this time.
        positions: ``{point_name: position}`` mapping each spatial
            point name (e.g. ``"x"``, ``"y"``) to a scalar spatial
            coordinate.  Points not in the dict default to 0.
        cache: A :class:`PropagatorCache` with pre-computed *C* table.
        t_min: Lower time bound (default 0).
        n_samples: Number of Sobol samples (should be a power of 2).
        seed: Random seed for the Sobol sequence.

    Returns:
        ``(estimate, std_error)`` for the summed two-point function.
    """
    from scipy.stats import qmc as _qmc

    total = 0.0
    total_err_sq = 0.0

    for ig in integrands:
        sp = ig.spatial
        ivs = list(reversed(sp.time_integration_vars))
        evs = list(sp.external_points)

        # Build direction dict from positions.
        # External points (which carry user-specified positions) take
        # priority over vertex points (which default to 0).
        directions: dict[str, float] = {}
        for pt in evs:
            dvar = sp.direction_map.get(pt)
            if dvar is not None and pt in positions:
                directions[dvar] = positions[pt]
        for pt, dvar in sp.direction_map.items():
            if dvar not in directions:
                directions[dvar] = positions.get(pt, 0.0)

        # Pre-compute spatial factors for each C propagator
        # factor = kappa2(n1, t_ref, n2, t_ref) / kappa2(0, t_ref, 0, t_ref)
        t_ref = max(t_min + 0.1, t_f * 0.5)
        model = cache.model
        kappa_00 = model.kappa2(
            np.array([0.0]), t_ref, np.array([0.0]), t_ref
        )  # (N, N)
        c_spatial_factors = []
        for sp_l, sp_r, _il, _ir in sp.c_propagators:
            dir_l = sp.direction_map[sp_l]
            dir_r = sp.direction_map[sp_r]
            n_l = directions.get(dir_l, 0.0)
            n_r = directions.get(dir_r, 0.0)
            if abs(n_l - n_r) < 1e-15:
                c_spatial_factors.append(np.ones(model.n_components))
            else:
                kappa_sep = model.kappa2(
                    np.array([n_l]), t_ref, np.array([n_r]), t_ref
                )
                # Diagonal ratio per component
                diag_00 = np.diag(kappa_00)
                diag_sep = np.diag(kappa_sep)
                safe = np.abs(diag_00) > 1e-30
                factor = np.where(safe, diag_sep / diag_00, 0.0)
                c_spatial_factors.append(factor)

        # --- Zero integration variables: direct evaluation ---
        et = {v: t_f for v in evs}
        ni = len(ivs)
        if ni == 0:
            val = ig.evaluate(et, directions, cache)
            total += val.real
            continue

        # --- Causal parent map ---
        pm: dict[str, list[str]] = defaultdict(list)
        for earlier, later in sp.time_orderings:
            if earlier in ivs:
                pm[earlier].append(later)

        # --- Sobol samples ---
        u = _qmc.Sobol(d=ni, seed=seed).random(n_samples)
        t_s = np.zeros((n_samples, ni))
        jac = np.ones(n_samples)
        for k, var in enumerate(ivs):
            ps = pm.get(var, [])
            pi = [ivs.index(p) for p in ps if p in ivs]
            ep = [t_f for p in ps if p not in ivs]
            hi = np.min(t_s[:, pi], axis=1) if pi else np.full(n_samples, t_f)
            if ep:
                hi = np.minimum(hi, min(ep))
            w = hi - t_min
            ok = w > 0
            t_s[:, k] = np.where(ok, t_min + u[:, k] * w, t_min)
            jac = np.where(ok, jac * w, 0.0)

        # --- Full time array ---
        all_vars = evs + ivs
        var_col = {v: j for j, v in enumerate(all_vars)}
        t_arr = np.empty((n_samples, len(all_vars)))
        for j in range(len(evs)):
            t_arr[:, j] = t_f
        for j in range(ni):
            t_arr[:, len(evs) + j] = t_s[:, j]

        # --- Vectorised R product ---
        r_prod = np.ones(n_samples)
        for sl, sr in sp.r_propagators:
            r_prod *= cache.R_time_batch(
                t_arr[:, var_col[sl]], t_arr[:, var_col[sr]]
            )

        # --- Vectorised C product with spatial factors ---
        dt = ig.diagram_term
        coeff = ig.coupling_array
        prop_idx = dt.propagator_indices

        # Fixed indices from the integrand (e.g. observable component
        # indices like {'a': 0, 'b': 1}).  These must participate in
        # _resolve_component so that C-propagator legs carrying a fixed
        # index name are evaluated at the correct component instead of
        # being summed over all components.
        fi = ig.fixed_indices

        if not prop_idx:
            # Scalar coupling (iso_R + iso_C)
            c_prod = np.ones(n_samples)
            for ci, (sp_l, sp_r, il, ir) in enumerate(sp.c_propagators):
                t_l = t_arr[:, var_col[sp_l]]
                t_r = t_arr[:, var_col[sp_r]]
                C_diag = cache.C_diagonal_batch(t_l, t_r)
                sf = c_spatial_factors[ci]
                a = DiagramIntegrand._resolve_component(il, fi)
                if a is not None:
                    c_prod *= C_diag[:, a] * sf[a]
                else:
                    c_prod *= (C_diag * sf[np.newaxis, :]).sum(axis=1)
            values = r_prod * complex(coeff).real * c_prod * jac

        else:
            # Propagator-indexed coupling
            idx_names = [name for name, _ in prop_idx]
            prop_shape = tuple(dim for _, dim in prop_idx)
            values = np.zeros(n_samples)
            for pidx in np.ndindex(*prop_shape):
                c_val = (
                    complex(coeff[pidx]).real
                    if coeff.ndim > 0
                    else complex(coeff).real
                )
                if abs(c_val) < 1e-20:
                    continue
                idx_map = {**fi, **dict(zip(idx_names, pidx))}
                c_prod = np.ones(n_samples)
                for ci, (sp_l, sp_r, il, ir) in enumerate(
                    sp.c_propagators
                ):
                    t_l = t_arr[:, var_col[sp_l]]
                    t_r = t_arr[:, var_col[sp_r]]
                    C_diag = cache.C_diagonal_batch(t_l, t_r)
                    sf = c_spatial_factors[ci]
                    a = DiagramIntegrand._resolve_component(il, idx_map)
                    b = DiagramIntegrand._resolve_component(ir, idx_map)
                    if a is not None and b is not None:
                        c_prod *= C_diag[:, a] * sf[a]
                    else:
                        c_prod *= (C_diag * sf[np.newaxis, :]).sum(axis=1)
                values += c_val * r_prod * c_prod * jac

        values = np.where(jac > 0, values, 0.0)
        est = float(np.mean(values))
        total += est

        # Error from 8 sub-batches
        bs = n_samples // 8
        bm = np.array(
            [np.mean(values[j * bs : (j + 1) * bs]) for j in range(8)]
        )
        total_err_sq += (float(np.std(bm, ddof=1) / np.sqrt(8))) ** 2

    return (total, np.sqrt(total_err_sq))
