"""Numerical evaluation pipeline for DiagramTerm objects.

Provides the 4-step workflow:
1. Coupling coefficients — handled by DiagramTerm.evaluate_coupling() (existing)
2. Spatial structure analysis — analyze_spatial()
3. Propagator evaluation — PropagatorModel + PropagatorCache
4. Contraction & integration — DiagramIntegrand
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

import hashlib

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
    #: For an ``equal_time`` non-local vertex, only the canonical
    #: representative leg appears here; the other (m-1) legs of that
    #: vertex are removed and their time values are read through the
    #: ``equal_time_aliases`` map below.
    time_integration_vars: tuple[str, ...]

    #: Surviving direction integration variables (one per R-component that
    #: contains only integration points — typically empty).
    direction_integration_vars: tuple[str, ...]

    #: External (non-integration) spatial points.
    external_points: tuple[str, ...]

    #: Maps non-representative internal spatial labels to their canonical
    #: time representative, propagated from
    #: ``DiagramTerm.equal_time_aliases``. The integration loops look up
    #: every leg's time via ``equal_time_aliases.get(label, label)`` so
    #: aliased legs share a single integration variable while keeping
    #: independent spatial labels. Empty tuple ⇒ no aliasing (original
    #: cross-spacetime cumulant behaviour).
    equal_time_aliases: tuple[tuple[str, str], ...] = ()

    #: ``(partner_label, leg_label)`` pairs identifying R-propagators
    #: whose factor has been absorbed into an upstream
    #: ``NonLocalVertex(already_R_contracted=True)`` callable.
    #: Propagated straight from ``DiagramTerm.r_absorbed_pairs``. The
    #: integrand R-product loop iterates over this set to **skip** the
    #: matching propagators (so the factor becomes 1 rather than
    #: ``R(t, t) = 0`` under Itô after the leg time aliases onto the
    #: partner's). The leg's time / direction collapse onto the
    #: partner's via the accompanying ``equal_time_aliases`` entries.
    r_absorbed_pairs: tuple[tuple[str, str], ...] = ()


def _kept_r_propagators(
    spatial: "SpatialStructure",
) -> tuple[tuple[str, str], ...]:
    """Return ``spatial.r_propagators`` minus any pair listed in
    ``spatial.r_absorbed_pairs``.

    Absorbed R-propagators contribute to direction grouping and time
    ordering (the leg label is union-find'd with its partner via the
    original R-propagator) but their R-factor has been folded into an
    upstream ``κ^(m)_R`` callable. The integrand R-product loops must
    skip them; calling this helper centralises the filter so every
    factor-multiplication site applies the same rule.
    """
    if not spatial.r_absorbed_pairs:
        return spatial.r_propagators
    absorbed = set(spatial.r_absorbed_pairs)
    return tuple(p for p in spatial.r_propagators if p not in absorbed)


def _select_C_batch(
    C_batch: np.ndarray,
    a: int | None,
    b: int | None,
) -> np.ndarray:
    """Select a per-sample C component from either diagonal or full batches."""
    C_arr = np.asarray(C_batch)
    if C_arr.ndim == 3:
        if a is not None and b is not None:
            return C_arr[:, a, b]
        return np.einsum("iaa->i", C_arr)
    if C_arr.ndim != 2:
        raise ValueError(
            f"C batch must have shape (n, N) or (n, N, N); got {C_arr.shape}."
        )
    if a is not None and b is not None:
        if a != b:
            return np.zeros(C_arr.shape[0], dtype=C_arr.dtype)
        return C_arr[:, a]
    return C_arr.sum(axis=1)


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
    # Equal-time non-local vertices share a single time integration
    # variable across their m legs. Build the alias map (leg → canonical
    # representative) and exclude the non-representatives from the
    # surviving time-integration set so the Jacobian is one factor of
    # ``width`` per vertex, not ``width^m``. Spatial labels remain
    # independent (each leg keeps its own direction / position).
    equal_time_aliases = dict(getattr(dt, "equal_time_aliases", ()) or ())
    time_integration_set = set(integration_set)
    for non_rep in equal_time_aliases:
        time_integration_set.discard(non_rep)

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
    # Only the canonical equal-time representatives (and any non-equal-time
    # internals) appear in time_integration_set; the topo sort drops the
    # filtered non-representatives implicitly because they're absent from
    # the input list and from any (earlier, later) entry whose endpoint is
    # filtered (rewritten below). Time-orderings that touch a filtered
    # label are remapped to its canonical representative so causality
    # between an equal-time vertex and the rest of the diagram remains
    # captured by R-propagators on the surviving (representative) legs.
    time_orderings_collapsed: list[tuple[str, str]] = []
    for earlier, later in time_orderings:
        earlier_alias = equal_time_aliases.get(earlier, earlier)
        later_alias = equal_time_aliases.get(later, later)
        if earlier_alias == later_alias:
            # Same physical time after collapse — no ordering needed.
            continue
        time_orderings_collapsed.append((earlier_alias, later_alias))
    sorted_time_vars = _topological_sort_times(
        tuple(sorted(time_integration_set)), time_orderings_collapsed
    )

    # --- External points ---
    external_points = tuple(sorted(p for p in all_points if p not in integration_set))

    return SpatialStructure(
        direction_groups=direction_groups,
        direction_map=direction_map,
        time_orderings=tuple(time_orderings_collapsed),
        r_propagators=tuple(r_props),
        c_propagators=tuple(c_props),
        time_integration_vars=tuple(sorted_time_vars),
        direction_integration_vars=tuple(sorted(direction_integration_vars)),
        external_points=external_points,
        equal_time_aliases=tuple(sorted(equal_time_aliases.items())),
        r_absorbed_pairs=tuple(getattr(dt, "r_absorbed_pairs", ()) or ()),
    )


# ---------------------------------------------------------------------------
# Step 3: Propagator Model and Cache
# ---------------------------------------------------------------------------


#: Relative tolerance used when projecting a diagram value onto the reals.
#: Two times are treated as ON the C-table diagonal when they differ by no
#: more than this, relatively.  C(t1,t2) has a derivative kink of exactly
#: -sigma2(t) on t1 == t2 (the integral's upper limit is min(t1,t2)), so the
#: substitution error from using the diagonal spline is bounded by
#: sigma2 * _DIAG_TOL -- orders below the table's own accuracy, hence never
#: the worse choice.
#: Entries kept in ``PropagatorCache``'s ``C_value`` memo before the oldest is
#: evicted.  It used to be unbounded -- described in comments as an LRU while
#: being a plain dict -- so a long sweep grew it without limit.
_C_CACHE_MAXSIZE = 65536

#: Two times are treated as ON the C-table diagonal when they differ by no
#: more than this, relatively.
_DIAG_TOL = 1e-9

_REALITY_TOL = 1e-9
#: Values below this magnitude are treated as an exact zero (a diagram that
#: cancelled), so a denormal-scale imaginary residue never raises.
_REALITY_FLOOR = 1e-290


def _real_or_raise(value, e_psi: int = 0, *, scale: float = 0.0,
                   where: str = "") -> float:
    """Project a diagram value onto the reals **without guessing a sign**.

    By the MSR reality theorem (see
    :meth:`~sft_wick.perturbation.DiagramTerm.observable_phase_factor`) a
    diagram equals ``i**(-E_psi)`` times a real number, provided each vertex
    coupling carries the ``(+/-i)**n_psi`` factor demanded by
    ``<phi psi> = -i R``.  Rotating by ``i**E_psi`` therefore lands exactly on
    the real axis.

    Anything left over is a mis-specified action -- most often a missing MSR
    factor on a multi-psi vertex, for which the required coefficient is
    ``-(i**m)/m!`` (see ``NonLocalVertex.msr_factor``).  That is reported, not
    silently turned into ``abs()`` (which flips the sign of negative
    contributions) or into ``0`` (which reads as "no contribution").

    Args:
        value: raw complex diagram value.
        e_psi: number of external response legs of the observable.
        scale: magnitude of the largest summand contributing to ``value``, so
            the test stays well posed when the diagram cancels to near zero.
        where: short context string included in the error message.
    """
    z = complex(value) * (1j ** int(e_psi))
    re_, im_ = z.real, z.imag
    if im_ == 0.0 or abs(z) <= _REALITY_FLOOR:
        return re_
    if abs(im_) <= _REALITY_TOL * max(abs(re_), abs(scale)):
        return re_
    raise ValueError(
        f"Diagram integrand{where} evaluated to {value!r}; after the "
        f"i**E_psi rotation (E_psi={e_psi}) it is {z!r}, whose imaginary part "
        f"is not negligible at {_REALITY_TOL:g} relative tolerance. Each "
        f"vertex coupling must carry the (+/-i)**n_psi factor demanded by "
        f"<phi psi> = -i R; for an m-leg all-psi vertex that factor is "
        f"-(i**m)/m! (see NonLocalVertex.msr_factor). Passing a real coupling "
        f"where an imaginary one is required is the usual cause."
    )


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
        direct_kwargs: dict | None = None,
    ):
        self.parent = parent
        self.t_max = t_max
        self.n_grid_t = n_grid_t
        self.mode = mode
        self.round_decimals = round_decimals
        #: ``method`` / ``n_gauss`` forwarded to ``_C_value_direct`` for
        #: every grid cell (empty for dblquad / closed-form, so subclasses
        #: overriding the legacy 4-argument signature keep working).
        self.direct_kwargs: dict = dict(direct_kwargs or {})
        self.ts = np.linspace(parent.model.t_min, t_max, n_grid_t)
        self._splines_by_key: dict = {}
        #: Separable-translation shortcut (see :meth:`_build`): the
        #: ``(t1, t2)`` grids at zero separation divided by ``κ_x(0)``, built
        #: once and rescaled by ``κ_x(r)`` for every later ``r``.
        self._base_grids: list | None = None
        #: Number of times the full quadrature grid was actually built --
        #: exposed so tests and the cost estimator can count table builds.
        self.n_grid_builds: int = 0
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
        """Build 2-D ``(t1, t2)`` splines at this specific parameter value.

        Two shortcuts, both exact up to floating-point rounding:

        * **Separability** (translation mode, ``κ²(1, 2) = κ_t · κ_x(r)``,
          no white noise): ``C(r; t1, t2) = κ_x(r) · C(0; t1, t2)``, so the
          quadrature grid is built ONCE at zero separation and rescaled
          for every further ``r``.  The lazy cache used to redo the full
          ``n_grid_t²`` quadrature per distinct ``r`` -- 4× the work for a
          four-point moment, 12×+ in a positions sweep.
        * **Time symmetry** (``κ_t`` even, diagonal C): the grid is
          symmetric in ``(t1, t2)``, so only the upper triangle is
          evaluated and mirrored.

        Whether either applies is decided by the parent cache
        (:meth:`PropagatorCache._lazy_spatial_factor` /
        :meth:`PropagatorCache._c_time_symmetric`); a user-supplied
        closed form or C callable gets neither, since nothing is known
        about its structure.

        Parallelises across grid points when :attr:`n_jobs` ≠ 1, with a
        progress bar in either case.
        """
        x1_arr = np.asarray(x1_val)
        x2_arr = np.asarray(x2_val)
        grids = None
        if self.mode == "translation":
            factor_fn = self.parent._lazy_spatial_factor()
            if factor_fn is not None:
                grids = self._grids_from_base(x1_arr, x2_arr, factor_fn)
        if grids is None:
            grids = self._grids_by_quadrature(x1_arr, x2_arr)
        return self._splines_from_grids(grids)

    def _grids_from_base(self, x1_arr, x2_arr, factor_fn) -> list | None:
        """Rescale the zero-separation grids by ``κ_x(r) / κ_x(0)``.

        Returns ``None`` (caller falls back to quadrature) when
        ``κ_x(0)`` vanishes or is not finite, so the shortcut can never
        divide by zero.
        """
        diff = np.asarray(x1_arr, dtype=float) - np.asarray(x2_arr, dtype=float)
        r = float(abs(diff)) if diff.ndim == 0 else float(np.linalg.norm(diff))
        f_r = float(factor_fn(r))
        if self._base_grids is None:
            f0 = float(factor_fn(0.0))
            if not np.isfinite(f0) or f0 == 0.0:
                return None
            zero = np.zeros_like(np.asarray(x1_arr, dtype=float))
            self._base_grids = [
                g / f0 for g in self._grids_by_quadrature(zero, zero)
            ]
        return [f_r * g for g in self._base_grids]

    def _grids_by_quadrature(self, x1_arr, x2_arr) -> list:
        """The per-component ``(n_t, n_t)`` grids from ``_C_value_direct``."""
        from .progress import progress_map

        parent = self.parent
        N = parent.model.n_components
        ts = self.ts
        n_t = self.n_grid_t
        symmetric = parent._c_time_symmetric()

        tasks = [
            (i, j, ts[i], ts[j])
            for i in range(n_t)
            for j in range(n_t)
            if (not symmetric) or j >= i
        ]

        direct_kwargs = self.direct_kwargs

        def _point(args):
            i, j, t1, t2 = args
            C_mat = parent._C_value_direct(x1_arr, t1, x2_arr, t2, **direct_kwargs)
            return i, j, np.array([C_mat[a, a] for a in range(N)])

        label = parent._c_source_label()
        results = progress_map(
            _point, tasks, f"C table ({label})",
            n_jobs=self.n_jobs, unit="cell",
        )
        self.n_grid_builds += 1

        grids = [np.zeros((n_t, n_t)) for _ in range(N)]
        for i, j, cvec in results:
            for a in range(N):
                grids[a][i, j] = cvec[a]
                if symmetric:
                    grids[a][j, i] = cvec[a]
        return grids

    def _splines_from_grids(self, grids: list) -> list:
        from scipy.interpolate import RectBivariateSpline

        ts = self.ts
        # Wrap each 2-D spline with the diagonal spline harvested from the
        # SAME grid -- the lazy spatial path carried the identical kink as
        # the legacy table (measured 22.3% at n_grid_t=41, bit-for-bit the
        # same numbers), and this is the path examples/demo1 uses.
        return [
            _DiagAwareSpline(
                RectBivariateSpline(ts, ts, g),
                _diag_line_interp(ts, np.diag(g)),
            )
            for g in grids
        ]


class _DiagLineSpline:
    """Picklable 1-D cubic spline of a C-table diagonal.

    Two things are true at once and neither alone gives a usable answer:

    * ``scipy.interpolate.CubicSpline`` is NOT picklable (scipy 1.18:
      ``TypeError: cannot pickle 'module' object``), so a
      :class:`PropagatorCache` holding one cannot be sent through joblib or
      saved -- a regression against ``RectBivariateSpline``, which is.
    * ``RegularGridInterpolator(method='cubic')`` IS picklable but DIVERGES on
      this data as the grid refines: mid-cell relative error 4.0e-04 at
      n_grid=41 but 8.2e-03 at n_grid=321.  Swapping to it trades a
      serialisation bug for a numerical one.

    So keep the CubicSpline and make the *container* picklable: only the nodes
    and values are serialised, and the spline is rebuilt on first use.
    """

    __slots__ = ("_ts", "_v", "_cs")

    def __init__(self, ts, values):
        object.__setattr__(self, "_ts", np.asarray(ts, dtype=float))
        object.__setattr__(self, "_v", np.asarray(values, dtype=float))
        object.__setattr__(self, "_cs", None)

    def _spline(self):
        if self._cs is None:
            from scipy.interpolate import CubicSpline
            object.__setattr__(self, "_cs", CubicSpline(self._ts, self._v))
        return self._cs

    def __call__(self, t):
        return np.asarray(self._spline()(np.asarray(t, dtype=float)),
                          dtype=float)

    def __getstate__(self):
        return (self._ts, self._v)

    def __setstate__(self, state):
        object.__setattr__(self, "_ts", state[0])
        object.__setattr__(self, "_v", state[1])
        object.__setattr__(self, "_cs", None)


def _diag_line_interp(ts, values):
    """Picklable 1-D cubic interpolator for a C-table diagonal."""
    return _DiagLineSpline(ts, values)


def _diag_line_eval(itp, t):
    """Evaluate a :func:`_diag_line_interp` interpolator at scalar or array t."""
    return np.atleast_1d(itp(np.atleast_1d(np.asarray(t, dtype=float))))


def _diag_grid_interp(axes, grid, method):
    """Interpolator over ``(t, *extra)`` built from ``grid[i, i, ...]``.

    The first two axes of every spatial C table are the SAME time grid, so
    ``grid[i, i, ...]`` is the diagonal slice -- already computed, no extra
    quadrature.  See :class:`_DiagAwareGridInterp` for why it is needed.
    """
    from scipy.interpolate import RegularGridInterpolator

    ts = np.asarray(axes[0])
    idx = np.arange(len(ts))
    return RegularGridInterpolator(
        (ts,) + tuple(axes[2:]), grid[idx, idx],
        bounds_error=False, fill_value=None, method=method,
    )


class _DiagAwareGridInterp:
    """A ``(t1, t2, *extra)`` C interpolator that knows about the ridge.

    ``C`` has a derivative discontinuity of exactly ``-sigma2(t)`` on
    ``t1 == t2`` (the integral's upper limit is ``min(t1, t2)``).  Any
    tensor-product rule blends across it: a cell straddling the diagonal mixes
    corners from both sides, so the ridge gets cut off.  Every tadpole
    evaluates ``C(s, s)``, exactly on it.

    This pairs the full interpolator with one built from the SAME grid's
    diagonal slice and routes numerically-equal times to it.  ``t -> C(t,t)``
    is smooth, so the diagonal interpolator has no ridge to resolve.

    ``__call__`` matches ``RegularGridInterpolator.__call__`` -- an ``(n, ndim)``
    array of points -- so it drops into every call site unchanged.
    """

    __slots__ = ("_full", "_diag")

    def __init__(self, full, diag):
        self._full = full
        self._diag = diag

    def __call__(self, pts):
        pts = np.asarray(pts, dtype=float)
        out = np.asarray(self._full(pts), dtype=float)
        if self._diag is None or pts.ndim < 2 or pts.shape[-1] < 2:
            return out
        t1, t2 = pts[..., 0], pts[..., 1]
        on = np.abs(t1 - t2) <= _DIAG_TOL * np.maximum(
            1.0, np.maximum(np.abs(t1), np.abs(t2))
        )
        if not np.any(on):
            return out
        td = 0.5 * (t1 + t2)
        dpts = np.concatenate([td[..., None], pts[..., 2:]], axis=-1)
        out = out.copy()
        out[on] = np.asarray(self._diag(dpts[on]), dtype=float)
        return out


    # Pickle: ``__slots__`` plus ``__getattr__`` delegation would otherwise
    # recurse forever on unpickling, when the slots are not yet set.
    def __getstate__(self):
        return {n: getattr(self, n) for n in self.__slots__}

    def __setstate__(self, state):
        for n, v in state.items():
            object.__setattr__(self, n, v)

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self._full, name)


class _DiagAwareSpline:
    """A 2-D ``(t1, t2)`` C-spline that knows about the diagonal kink.

    ``C(t1,t2) = int_0^{min(t1,t2)} R(t1,l) sigma2(l) R(t2,l) dl`` has a
    derivative discontinuity of exactly ``-sigma2(t)`` on ``t1 == t2``, which a
    tensor-product spline (C^2 by construction) cannot represent: it stops
    converging there while staying clean O(h^4) off the diagonal.  Every
    tadpole evaluates ``C(s,s)``, exactly on the kink.

    This wraps the 2-D spline together with a 1-D spline of the SAME grid's
    ``i == j`` entries -- no extra quadrature -- and routes equal times to it.
    ``t -> C(t,t)`` is smooth, so the 1-D spline restores O(h^4).

    The call signature matches ``RectBivariateSpline.__call__`` exactly, so it
    drops into every existing call site unchanged.
    """

    __slots__ = ("_s2d", "_sdiag")

    def __init__(self, s2d, sdiag):
        self._s2d = s2d
        self._sdiag = sdiag

    def __call__(self, t1, t2, grid=False):
        out = self._s2d(t1, t2, grid=grid)
        if self._sdiag is None or grid:
            return out
        t1a = np.asarray(t1, dtype=float)
        t2a = np.asarray(t2, dtype=float)
        on_diag = np.abs(t1a - t2a) <= _DIAG_TOL * np.maximum(
            1.0, np.maximum(np.abs(t1a), np.abs(t2a))
        )
        if not np.any(on_diag):
            return out
        td = 0.5 * (t1a + t2a)
        out = np.asarray(out, dtype=float)
        if out.ndim == 0:
            return _diag_line_eval(self._sdiag, td)[0]
        out = out.copy()
        out[on_diag] = _diag_line_eval(self._sdiag, np.atleast_1d(td)[on_diag])
        return out


    # Pickle: ``__slots__`` plus ``__getattr__`` delegation would otherwise
    # recurse forever on unpickling, when the slots are not yet set.
    def __getstate__(self):
        return {n: getattr(self, n) for n in self.__slots__}

    def __setstate__(self, state):
        for n, v in state.items():
            object.__setattr__(self, n, v)

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self._s2d, name)


def _C_value_direct_gl(
    model: "PropagatorModel",
    n1: Any,
    t1: float,
    n2: Any,
    t2: float,
    n_gauss: int = 20,
) -> np.ndarray:
    """Gauss-Legendre evaluation of the C-propagator 2-D integral.

    Drop-in replacement for the ``scipy.integrate.dblquad`` path inside
    :meth:`PropagatorCache._C_value_direct` for the common case where
    κ² is smooth except for a folded ``|λ1 − λ2|`` cusp on the diagonal
    (the OU / exponential-temporal kernels used in demo1, demo2, and
    the test suite).

    Strategy: rather than apply a single tensor-product Gauss-Legendre
    rule on the rectangle ``[t_min, t1] × [t_min, t2]`` -- where the
    diagonal cusp ruins polynomial convergence and gives O(10%) error
    even at ``n_gauss=30`` -- we split the rectangle at ``λ1 = λ2``
    into 2 (square case) or 3 (rectangle case) sub-regions on each of
    which the integrand is smooth, then apply a fixed
    ``n_gauss × n_gauss`` rule per sub-region.

    Sub-regions for ``t_min ≤ λ1 ≤ t1, t_min ≤ λ2 ≤ t2``:

    * Let ``t_d = min(t1, t2)``.  In the square ``[t_min, t_d]²`` the
      diagonal cuts through.

      - Region A (lower triangle): ``λ2 ≤ λ1``, so ``|λ1−λ2|=λ1−λ2``.
      - Region B (upper triangle): ``λ1 ≤ λ2``, so ``|λ1−λ2|=λ2−λ1``.

    * Outside the square (only if ``t1 ≠ t2``):

      - If ``t1 > t2``: Region C is the strip
        ``t_d ≤ λ1 ≤ t1, t_min ≤ λ2 ≤ t2`` where ``λ1 > λ2``.
      - If ``t2 > t1``: Region C is the strip
        ``t_min ≤ λ1 ≤ t1, t_d ≤ λ2 ≤ t2`` where ``λ2 > λ1``.

    Each sub-region is handled by an iterated 1-D Gauss-Legendre rule:
    outer pass over ``λ1`` with ``n_gauss`` nodes, inner pass over
    ``λ2`` with ``n_gauss`` nodes whose interval depends on ``λ1`` for
    the triangular regions.

    For the triangular regions the inner integral has a node-dependent
    upper or lower bound; we map the inner ``[-1, 1]`` GL nodes onto
    that variable interval each time, accumulating the Jacobian.

    The white-noise δ-correlated piece (when ``model.sigma2`` is set)
    is added via a single 1-D Gauss-Legendre pass on
    ``[t_min, min(t1, t2)]`` -- the integrand there is smooth (no cusp).

    Args:
        model: The propagator model.
        n1, t1, n2, t2: As in :meth:`PropagatorCache._C_value_direct`.
        n_gauss: Number of GL nodes per dimension per sub-region.

    Returns:
        ``(N, N)`` array of C values, identical in shape and meaning
        to the dblquad path.
    """
    from numpy.polynomial.legendre import leggauss

    N = model.n_components
    t_min = model.t_min
    C_mat = np.zeros((N, N))

    if t1 <= t_min or t2 <= t_min:
        # Empty integration domain along at least one axis -- C = 0.
        return C_mat

    # 1-D GL nodes & weights on [-1, 1], reused across sub-regions.
    nodes, weights = leggauss(n_gauss)

    # ---- helper: evaluate the R matrix at a single time pair ------ #
    def _r_mat(t_obs: float, lam: float) -> np.ndarray:
        """Return the full ``(N, N)`` response matrix ``R(t_obs, lam)``.

        For ``iso_R`` this is ``R_time * I``.  Otherwise ``R_time`` is
        expected to return the full matrix; note it is **not** in general
        diagonal in component indices — that holds only when the linear
        operator itself is diagonal in the chosen basis.  Using only the
        diagonal here silently corrupts C for any dense drift matrix
        (e.g. ``A = H + lambda`` with ``H = X^T X / N``).
        """
        rt = model.R_time(t_obs, lam)
        if model.iso_R:
            return float(rt) * np.eye(N)
        return np.asarray(rt, dtype=float)

    # ---- helper: evaluate κ² at a single time pair --------------- #
    def _kappa(lam1: float, lam2: float) -> np.ndarray:
        return np.asarray(model.kappa2(n1, lam1, n2, lam2), dtype=float)

    # ---- helper: contract one (R(t1,lam1), kappa, R(t2,lam2)) ----- #
    # triple into the (N, N) per-point integrand value.
    def _integrand_block(
        r1: np.ndarray, kmat: np.ndarray, r2: np.ndarray,
    ) -> np.ndarray:
        # C_{ab} = sum_{c,d} R_{ac}(t1,l1) kappa_{cd}(l1,l2) R_{bd}(t2,l2)
        #        = [ R1 @ kappa @ R2^T ]_{ab}
        # The transpose is essential and invisible for symmetric R --
        # do not "simplify" it away.
        out = r1 @ np.asarray(kmat, dtype=float) @ r2.T
        if model.diag_C:
            # Project onto the diagonal AFTER contracting, never before.
            return np.diag(np.diag(out))
        return out

    # ---- helper: integrate over a triangular region -------------- #
    # Region of the form  a_lo ≤ λ1 ≤ a_hi,  inner_lo(λ1) ≤ λ2 ≤ inner_hi(λ1).
    # Both ``a_lo, a_hi`` are scalars; ``inner_lo`` and ``inner_hi``
    # are callables of λ1.
    def _gl_region(
        lam1_lo: float, lam1_hi: float,
        inner_lo: Callable[[float], float],
        inner_hi: Callable[[float], float],
    ) -> np.ndarray:
        if lam1_hi <= lam1_lo:
            return np.zeros((N, N))
        # Outer mapping: λ1 = lam1_lo + 0.5*(lam1_hi-lam1_lo)*(node+1)
        half_outer = 0.5 * (lam1_hi - lam1_lo)
        mid_outer = 0.5 * (lam1_hi + lam1_lo)
        lam1_pts = mid_outer + half_outer * nodes
        outer_w = half_outer * weights

        acc = np.zeros((N, N))
        for k, lam1 in enumerate(lam1_pts):
            lo = inner_lo(float(lam1))
            hi = inner_hi(float(lam1))
            if hi <= lo:
                continue
            half_inner = 0.5 * (hi - lo)
            mid_inner = 0.5 * (hi + lo)
            lam2_pts = mid_inner + half_inner * nodes
            inner_w = half_inner * weights

            r1m = _r_mat(t1, float(lam1))
            inner_acc = np.zeros((N, N))
            for j, lam2 in enumerate(lam2_pts):
                kmat = _kappa(float(lam1), float(lam2))
                r2m = _r_mat(t2, float(lam2))
                inner_acc += inner_w[j] * _integrand_block(r1m, kmat, r2m)
            acc += outer_w[k] * inner_acc
        return acc

    t_d = min(t1, t2)

    # Region A: t_min ≤ λ2 ≤ λ1 ≤ t_d  (lower triangle, λ1 > λ2).
    if t_d > t_min:
        C_mat += _gl_region(
            t_min, t_d,
            inner_lo=lambda _l1: t_min,
            inner_hi=lambda l1: l1,
        )
        # Region B: t_min ≤ λ1 ≤ λ2 ≤ t_d  (upper triangle, λ1 < λ2).
        C_mat += _gl_region(
            t_min, t_d,
            inner_lo=lambda l1: l1,
            inner_hi=lambda _l1: t_d,
        )

    # Region C: outside the square (only if t1 != t2).
    if t1 > t2:
        # λ1 ∈ [t_d, t1], λ2 ∈ [t_min, t2] = [t_min, t_d].  λ1 > λ2.
        C_mat += _gl_region(
            t_d, t1,
            inner_lo=lambda _l1: t_min,
            inner_hi=lambda _l1: t_d,
        )
    elif t2 > t1:
        # λ2 ∈ [t_d, t2], λ1 ∈ [t_min, t1] = [t_min, t_d].  λ2 > λ1.
        C_mat += _gl_region(
            t_min, t_d,
            inner_lo=lambda _l1: t_d,
            inner_hi=lambda _l1: t2,
        )

    # ---- White-noise δ piece: 1-D GL on [t_min, min(t1, t2)] ----- #
    if model.sigma2 is not None:
        t_upper = t_d
        if t_upper > t_min:
            half = 0.5 * (t_upper - t_min)
            mid = 0.5 * (t_upper + t_min)
            tau_pts = mid + half * nodes
            tau_w = half * weights
            for k, tau in enumerate(tau_pts):
                tau_f = float(tau)
                r1m = _r_mat(t1, tau_f)
                r2m = _r_mat(t2, tau_f)
                sig = np.asarray(model.sigma2(n1, tau_f, n2), dtype=float)
                C_mat += tau_w[k] * _integrand_block(r1m, sig, r2m)

    return C_mat


def _probe_diagonal_cusp(model: "PropagatorModel", positions: tuple) -> bool:
    """Numerically decide whether ``κ²`` has a derivative jump at ``λ1 = λ2``.

    ``J(h) = |κ(λ, λ+h) + κ(λ, λ−h) − 2 κ(λ, λ)| / h`` is the jump of the
    one-sided derivatives.  For a smooth kernel it is ``O(h)``; for a
    ``|Δ|`` cusp it tends to a constant.  Compare ``h`` and ``h / 4``.
    """
    n1, n2 = positions
    t_min = float(model.t_min)
    try:
        vals = []
        for lam0 in (t_min + 0.7, t_min + 2.3):
            for h in (1e-3, 2.5e-4):
                k0 = np.asarray(model.kappa2(n1, lam0, n2, lam0), dtype=float)
                kp = np.asarray(model.kappa2(n1, lam0, n2, lam0 + h), dtype=float)
                km = np.asarray(model.kappa2(n1, lam0, n2, lam0 - h), dtype=float)
                scale = max(float(np.max(np.abs(k0))), 1e-300)
                vals.append(float(np.max(np.abs(kp + km - 2.0 * k0))) / h / scale)
    except Exception:
        return True
    j_h, j_h4 = vals[0] + vals[2], vals[1] + vals[3]
    if j_h4 < 1e-9:
        return False
    return bool(j_h4 > 0.5 * j_h)


def select_gl_node_count(
    model: "PropagatorModel",
    t_max: float,
    *,
    n_start: int = 20,
    n_max: int = 110,
    tol: float = 1e-8,
    growth: float = 1.5,
    positions: tuple | None = None,
) -> int | None:
    """Smallest Gauss-Legendre node count that is converged for a table
    reaching ``t_max``, or ``None`` if none up to ``n_max`` is.

    A fixed-node tensor-product rule loses accuracy as the integrand's
    exponential rates times the interval length grow -- for the demo1
    kernel (``γ = 1``, ``σ_t = 0.3``) ``n_gauss = 20`` is at 1e-12 for
    ``t_max = 15`` but 1e-5 at 30 and 2e-2 at 100.  Rather than trust a
    number, this probes the table's extreme cells (the deep corner
    ``(t_max, t_max)``, its mid-table neighbours and the thinnest strip
    ``(t_max, t_min + h)``) and accepts ``n`` when the rule agrees with
    its ``growth × n`` refinement to ``tol`` relative at every probe
    (cells below ``1e-6`` of the table scale are judged against that
    floor).  This is the nested-rule convergence test every adaptive
    quadrature runs, applied once per table instead of per cell; it
    costs a handful of GL evaluations.

    Returns ``None`` -- meaning "use dblquad" -- when the cap is reached
    or the kernel raises on the probe positions.
    """
    import math

    t_min = float(model.t_min)
    if float(t_max) <= t_min:
        return int(n_start)
    if positions is None:
        n1 = n2 = np.asarray(0.0)
    else:
        n1, n2 = positions
    span = float(t_max) - t_min
    h = span / 60.0
    mid = t_min + 0.5 * span
    cells = [
        (t_max, t_max), (t_max, mid), (mid, mid),
        (t_max, t_min + h), (t_min + h, t_min + h),
    ]
    n = int(n_start)
    try:
        while True:
            n_ref = max(n + 2, int(math.ceil(growth * n)))
            if n_ref > n_max:
                return None
            vals = [_C_value_direct_gl(model, n1, t1, n2, t2, n_gauss=n)
                    for t1, t2 in cells]
            refs = [_C_value_direct_gl(model, n1, t1, n2, t2, n_gauss=n_ref)
                    for t1, t2 in cells]
            scale = max(float(np.max(np.abs(r))) for r in refs)
            if not np.isfinite(scale) or scale == 0.0:
                return None
            worst = max(
                float(np.max(np.abs(v - r) / np.maximum(np.abs(r), 1e-6 * scale)))
                for v, r in zip(vals, refs)
            )
            if worst <= tol:
                return n
            n = n_ref
    except Exception:
        return None


def _causal_lower_bounds(
    spatial,
    int_vars,
    external_times: dict,
    t_min: float,
) -> dict:
    """Lower limits imposed on internal times by *external* response legs.

    A time ordering ``(earlier, later)`` from an R propagator can be expressed
    as an **upper** bound only when ``earlier`` is itself an integration
    variable — that is the form every bound-builder in this module uses::

        for earlier, later in spatial.time_orderings:
            if earlier in int_vars:
                upper_bounds[earlier].append(later)

    The mirrored case is silently dropped by that filter: when ``earlier`` is an
    **external** point and ``later`` is internal, the constraint
    ``t_later >= t_earlier`` is a **lower** bound and has nowhere to go.  That
    case arises whenever the observable itself carries a response leg, since
    ``<phi(u) psi(y)> = R(u, y)`` forces ``t_u >= t_y``.  For the order-1
    response function of a quartic theory the orderings are
    ``(('y','y_0'), ('y_0','x'))`` and only the second was ever applied, so the
    vertex time was integrated over ``[t_min, t_x]`` instead of ``[t_y, t_x]``.

    Returns ``{int_var: lower_limit}`` for every internal variable that has at
    least one external lower bound; ``t_min`` is folded in, so the caller can
    use the value directly as ``lo``.
    """
    return _causal_lower_bound_sources(
        spatial, int_vars, external_times, t_min,
    )[0]


def _causal_lower_bound_sources(
    spatial,
    int_vars,
    external_times: dict,
    t_min: float,
    swept: tuple = (),
) -> tuple[dict, dict]:
    """Lower bounds split into a constant part and a *variable* part.

    ``external_times`` maps each **fixed** external point to its time, and
    ``swept`` names the externals the caller is sampling (``integrate_over``).
    An ordering from a swept external is still a lower bound, but its value
    is only known per sample, so it is reported as a *source name* rather
    than a number.

    Returns ``(const, sources)`` where

    * ``const[v]`` is the constant lower limit for internal variable ``v``
      (``t_min`` folded in), exactly what :func:`_causal_lower_bounds`
      returns, and
    * ``sources[v]`` is a tuple of names drawn from ``swept`` that also
      bound ``v`` from below, so the caller can take a per-sample
      ``max(const[v], *[t(s) for s in sources[v]])``.

    Only names in ``swept`` become sources.  A point that is in neither
    ``external_times`` nor ``swept`` -- an aliased leg of an ``equal_time``
    non-local vertex, say -- is skipped exactly as before, so the caller
    never receives a name it has no column for.

    Every sampler in this module draws the integrated externals *before*
    any internal variable, and ``nquad`` places them outside every internal
    variable in ``all_vars``, so a source's value is always available by
    the time the bound is needed.
    """
    int_set = set(int_vars)
    swept_set = set(swept)
    lowers: dict = {}
    sources: dict = {}
    # Direct constraints: (external earlier) -> (internal later).
    for earlier, later in spatial.time_orderings:
        if later in int_set and earlier not in int_set:
            t_e = external_times.get(earlier)
            if t_e is None:
                if earlier in swept_set:
                    sources.setdefault(later, set()).add(earlier)
            else:
                lowers[later] = max(lowers.get(later, t_min), float(t_e))

    # TRANSITIVE CLOSURE along internal edges.  A chain
    # ``ext -> v1 -> v2`` implies ``t_v2 >= t_v1 >= t_ext``, but only ``v1``
    # is named in a direct (external, internal) ordering.  Leaving ``v2`` at
    # ``t_min`` lets it range below ``t_ext``, at which point ``v1``'s interval
    # ``[t_ext, t_v2]`` inverts -- nquad then integrates backwards and returns
    # a negative volume.  The internal orderings form a DAG, so this fixpoint
    # terminates; it is cheap because diagrams have very few vertices.
    # Variable sources propagate along the same edges and for the same
    # reason.
    changed = True
    while changed:
        changed = False
        for earlier, later in spatial.time_orderings:
            if earlier in int_set and later in int_set:
                lo_e = lowers.get(earlier)
                if lo_e is not None and lo_e > lowers.get(later, t_min):
                    lowers[later] = lo_e
                    changed = True
                src_e = sources.get(earlier)
                if src_e and not src_e <= sources.get(later, set()):
                    sources.setdefault(later, set()).update(src_e)
                    changed = True
    return lowers, {v: tuple(sorted(s)) for v, s in sources.items()}


def _resolve_external_times(
    spatial, ext_fixed, lambda_f: float, external_times=None,
    integrate_over=(),
) -> tuple[dict, float]:
    """``{fixed external point -> its time}`` plus the global time ceiling.

    ``lambda_f`` plays two distinct roles in this module and only one of them
    generalises:

    * the **sweep limit** for an integrated external, and the default upper
      bound for an internal variable with no causal parent -- these stay tied
      to ``lambda_f`` (raised to the ceiling below when some external is
      pinned later than it), and
    * the **time of a fixed external point** -- which ``external_times`` now
      overrides per point.  Conflating the two is why unequal external times
      were unreachable: an observable such as ``R(t, t')`` needs its two legs
      at different times, and every production integrator pinned them both.

    ``external_times=None`` reproduces the old behaviour exactly.

    Returns ``(times, ceiling)`` where ``ceiling = max(lambda_f, *times)`` --
    an internal variable with no causal parent may still precede an external
    pinned beyond ``lambda_f``, so the default upper bound must cover it.

    Raises:
        ValueError: for an unknown point name, or for a point that is both
            pinned here and swept via ``integrate_over``.
    """
    if external_times:
        known = set(spatial.external_points)
        unknown = sorted(set(external_times) - known)
        if unknown:
            raise ValueError(
                f"external_times names unknown external point(s) {unknown}; "
                f"this diagram's external points are {sorted(known)}."
            )
        clash = sorted(set(external_times) & set(integrate_over or ()))
        if clash:
            raise ValueError(
                f"external point(s) {clash} appear in BOTH external_times and "
                f"integrate_over -- a point cannot be pinned at a fixed time "
                f"and swept at the same time."
            )
    et = external_times or {}
    times = {v: float(et.get(v, lambda_f)) for v in ext_fixed}
    ceiling = max([float(lambda_f), *times.values()]) if times else float(lambda_f)
    return times, ceiling


def _causal_reachability(time_orderings) -> dict:
    """``{node: set of nodes strictly later than it}``, transitively closed.

    Closing over the WHOLE ordering graph -- not just edges whose endpoints are
    of one kind -- is what makes the constraint table exhaustive.  Enumerating
    edge kinds by endpoint type ({fixed-ext, swept-ext, internal}^2) and
    handling each directly is how the branch previously missed three cells:
    swept -> internal -> swept, fixed-ext -> swept, and swept -> fixed-ext.
    """
    succ: dict = {}
    for e, l in time_orderings:
        succ.setdefault(e, set()).add(l)
        succ.setdefault(l, set())
    changed = True
    while changed:                      # the ordering graph is a small DAG
        changed = False
        for n in list(succ):
            grown = set(succ[n])
            for m in succ[n]:
                grown |= succ.get(m, set())
            if grown != succ[n]:
                succ[n] = grown
                changed = True
    return succ


def _swept_external_order(
    spatial, ext_integrated, fixed_times=None, lambda_f=None, t_min=0.0,
) -> tuple:
    """Causal constraints ON the swept externals themselves.

    A swept external is drawn freely over ``[t_min, lambda_f]`` unless some
    ordering constrains it.  Three kinds do, and none is carried by
    ``_causal_lower_bound_sources`` (which needs the LATER endpoint internal)
    or by ``parent_map`` / ``upper_bounds`` (which need the EARLIER one
    internal):

    * **swept -> swept**, directly or through internal vertices.  Both give
      ``t_later > t_earlier``; the mediated case is why this works off the
      transitive closure rather than the raw edge list.
    * **fixed-ext -> swept**: a constant lower bound at the fixed point's time.
    * **swept -> fixed-ext**: a constant upper bound at that time.

    Theta keeps the VALUE right in every case -- the integrand vanishes on the
    unconstrained part -- so these are quadrature concerns.  But they are the
    same jump-inside-the-domain that costs ``gauss_legendre`` its spectral
    convergence (measured 22% and 29% at the library default ``n_gauss=8``).

    Returns ``(order, lowers, const_lo, const_hi)``: the swept externals
    topologically sorted so every predecessor is drawn first, the swept
    predecessors of each, and its constant bounds.  With no constraint at all
    the order is the caller's own and the dicts are empty, so sampling stays
    bit-identical.
    """
    ext_set = set(ext_integrated)
    succ = _causal_reachability(tuple(spatial.time_orderings))
    fixed_times = fixed_times or {}

    lowers: dict = {}
    const_lo: dict = {}
    const_hi: dict = {}
    for v in ext_integrated:
        pre = [u for u in succ if v in succ.get(u, ()) and u != v]
        sw = [u for u in pre if u in ext_set]
        if sw:
            lowers[v] = sorted(sw)
        fx = [fixed_times[u] for u in pre if u in fixed_times]
        if fx:
            const_lo[v] = max([float(t_min)] + [float(t) for t in fx])
        post_fx = [fixed_times[u] for u in succ.get(v, ()) if u in fixed_times]
        if post_fx and lambda_f is not None:
            const_hi[v] = min([float(lambda_f)] + [float(t) for t in post_fx])

    if not lowers:
        return list(ext_integrated), {}, const_lo, const_hi

    # Kahn topological sort, stable in the caller's order.  A cycle cannot
    # arise from retarded R propagators; fall back rather than drop a variable.
    remaining = list(ext_integrated)
    placed: list = []
    placed_set: set = set()
    while remaining:
        ready = [v for v in remaining
                 if all(src in placed_set for src in lowers.get(v, ()))]
        if not ready:
            return list(ext_integrated), {}, const_lo, const_hi
        for v in ready:
            placed.append(v)
            placed_set.add(v)
            remaining.remove(v)
    return placed, lowers, const_lo, const_hi


def _real_batch_or_raise(values, e_psi: int = 0, *, where: str = "") -> np.ndarray:
    """Array counterpart of :func:`_real_or_raise`.

    The dynamic-coupling batch sites took a bare ``np.real(...)`` after the
    ``i**E_psi`` rotation, so a mis-specified action -- a real coupling where
    the MSR convention wants an imaginary one, say -- silently lost its
    imaginary part instead of being reported.  That is a hole in the reality
    projection on precisely the feature this branch adds.

    Vectorised so a per-sample batch is checked in one pass: the residue is
    judged against the batch's own scale, matching the scalar helper's
    ``max(|re|, scale)`` rule rather than element-by-element, which would
    reject legitimate near-zero samples inside an otherwise healthy batch.
    """
    z = np.asarray(values) * (1j ** int(e_psi))
    re_ = np.real(z)
    im_ = np.imag(z)
    if not np.any(im_):
        return re_
    scale = float(np.max(np.abs(re_))) if re_.size else 0.0
    worst = float(np.max(np.abs(im_))) if im_.size else 0.0
    if worst <= _REALITY_TOL * max(scale, _REALITY_FLOOR):
        return re_
    raise ValueError(
        f"Diagram integrand batch{where} has a non-negligible imaginary part "
        f"after the i**E_psi rotation (E_psi={e_psi}): worst |Im| = {worst:.3e} "
        f"against |Re| scale {scale:.3e}.  Check the coupling's phase "
        f"convention -- the MSR reality theorem gives i**(-E_psi) x real only "
        f"when each vertex coupling carries (+-i)**n_psi(v)."
    )


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
        c_method: str = "auto",
        n_gauss: int = 20,
        c_value_fn: Callable | None = None,
    ):
        """Initialise the propagator cache.

        Args:
            model: Propagator model specifying R and κ².
            quad_opts: Options forwarded to ``scipy.integrate.dblquad``.
            c_value_fn: Optional ``(n1, t1, n2, t2) -> (N, N)`` callable that
                **replaces** the quadrature construction of C entirely.

                By default C is *derived* from R and the noise cumulant as
                ``C = ∫∫ R κ R``.  That relation does not hold for every
                physically meaningful propagator pair — notably a
                disorder-averaged (DMFT) solution, where
                ``⟨R κ R⟩ ≠ ⟨R⟩ κ ⟨R⟩`` — so such a C must be supplied
                directly.  This is the public, L0 equivalent of the L1
                ``Propagators.build(c_closed_form=..., c_closed_form_only=True)``
                route; previously it required subclassing and overriding
                :meth:`_C_value_direct`.

                It is consulted by :meth:`_C_value_direct`, hence by
                :meth:`C_value`, :meth:`C_diagonal` and every
                ``precompute_C_table_*`` builder.
            c_method: Quadrature method used by :meth:`_C_value_direct`
                (and therefore by every ``precompute_C_table_*`` builder).

                - ``'auto'`` (default): Gauss-Legendre, with the node
                  count chosen at the first table build by
                  :func:`select_gl_node_count` -- the rule is refined
                  until it agrees with itself at the table's extreme
                  cells -- and ``'dblquad'`` as the fallback when no
                  node count up to the cap converges.  Direct
                  :meth:`C_value` calls made *before* any table build
                  use ``'dblquad'``, since no ``t_max`` is known yet.
                - ``'gauss_legendre'``: tensor-product Gauss-Legendre
                  with diagonal-aware sub-region splitting at
                  ``λ1 = λ2`` and exactly ``n_gauss`` nodes -- no
                  convergence check.  ~10-1000× faster than
                  ``'dblquad'`` on smooth κ² with a single ``|λ1−λ2|``
                  cusp on the diagonal (the demo1/demo2 OU-style
                  kernels).  See :func:`_C_value_direct_gl`.  Its
                  accuracy at fixed ``n_gauss`` degrades as
                  ``(γ + 1/σ_t) · t_max`` grows (measured: 1e-12 at
                  ``t_max = 15``, 1e-5 at 30, 2e-2 at 100 for the
                  demo1 kernel with ``n_gauss=20``), which is why
                  ``'auto'`` re-checks instead of trusting 20.
                - ``'dblquad'`` (robust): adaptive 2-D quadrature via
                  :func:`scipy.integrate.dblquad`.  Handles arbitrary
                  κ² but slow (10-250 ms per call) due to adaptive
                  recursion around the diagonal cusp.

                Each ``precompute_C_table_*`` builder also accepts a
                local ``c_method`` kwarg that overrides this default
                for the duration of the build.
            n_gauss: Number of Gauss-Legendre nodes per dimension per
                sub-region for ``c_method='gauss_legendre'``, and the
                STARTING node count for ``'auto'``.  Default ``20``.
                Cost scales as ``n_gauss²`` per sub-region (3
                sub-regions for ``t1 ≠ t2``, 2 for ``t1 = t2``).
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
        if c_method not in ("auto", "dblquad", "gauss_legendre"):
            raise ValueError(
                f"c_method must be 'auto', 'dblquad' or 'gauss_legendre'; "
                f"got {c_method!r}"
            )
        if int(n_gauss) < 2:
            raise ValueError(
                f"n_gauss must be >= 2; got {n_gauss!r}"
            )
        self.model = model
        self.quad_opts = quad_opts or {}
        self.homogeneity = homogeneity
        self.interp_method = interp_method
        self.c_method = c_method
        self.n_gauss = int(n_gauss)
        #: What ``'auto'`` resolved to at the first table build (``None``
        #: until then, or when ``c_method`` was explicit).
        self._c_method_resolved: str | None = None
        self._n_gauss_resolved: int | None = None
        #: Optional user-supplied C, bypassing the ``∫∫ R κ R`` construction.
        self.c_value_fn = c_value_fn

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

        #: Bounded LRU memo for ``C_value`` (``move_to_end`` on every hit,
        #: eviction from the cold end).  It was an UNBOUNDED plain dict
        #: described in comments as an LRU -- on a long sweep it grew without
        #: limit, which matters in a project whose review agents have already
        #: OOM'd once.  The bound is on entry COUNT: one entry is an
        #: ``(N, N)`` correlator, so the memory ceiling scales as N^2 and a
        #: large-N run should lower :data:`_C_CACHE_MAXSIZE` or call
        #: :meth:`clear_cache`.  Excluded from pickling -- see
        #: :meth:`__getstate__`.
        self._c_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        # Legacy 2-D (t1, t2) spline at fixed x — still populated by
        # ``precompute_C_table`` for backward compatibility.
        self._c_splines: list | None = None
        #: 1-D splines of C(t, t) harvested from the same grid -- see
        #: ``precompute_C_table``.  The 2-D tensor-product spline is C^2 by
        #: construction and cannot represent the diagonal kink.
        self._c_diag_splines: list | None = None
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
        """Evaluate R_time(t_left, t_right) — the RAW model accessor.

        Returns scalar if ``model.iso_R=True``, else ``(N, N)`` array.

        Θ is deliberately **not** applied here.  This is the single
        authoritative statement of the convention, which used to be
        described three inconsistent ways across this module:

        * ``R_time`` is Θ-stripped, so a model author writes only the
          retarded branch and never has to encode the step function.
        * Θ is applied at **diagram evaluation**, by every consumer that
          multiplies R factors together: :meth:`R_product`,
          :meth:`R_time_batch`, and
          ``DiagramIntegrand._evaluate_r_product_general``.  All three use
          the strict test ``t_left > t_right`` (Itô: equal times give 0),
          and none of them calls this method for an acausal pair.
        * The older claim that "the integration domain handles causality"
          holds only where an integration domain exists.  An R propagator
          joining two *fixed external* points has none — see
          :meth:`R_product`.

        Callers wanting the physical, Θ-enforced propagator should use
        :meth:`R_time_batch` (or evaluate a diagram), not this method.
        """
        return self.model.R_time(t_left, t_right)

    def R_product(
        self,
        r_pairs: tuple[tuple[str, str], ...],
        times: dict[str, float],
    ) -> float:
        """Product of R_time over all R propagator pairs.

        Only valid when ``model.iso_R=True`` (R is scalar).

        Retardation is enforced here.  ``R_time`` itself is the raw model
        accessor and deliberately does not apply Θ ("the integration domain
        handles causality") — but that only holds when there *is* an
        integration domain.  An R propagator joining two **fixed external**
        points has none: at order 0, ``<phi(t_x) psi(t_y)>`` would otherwise
        evaluate to the unbounded acausal ``exp(+mu (t_y - t_x))`` for
        ``t_y > t_x``.  Under the Itô prescription equal times give 0 as well.
        """
        result = 1.0
        for sl, sr in r_pairs:
            t_l, t_r = times[sl], times[sr]
            if not t_l > t_r:
                return 0.0
            result *= float(self.R_time(t_l, t_r))
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

        # Check the (n1, t1, n2, t2) memo
        # Key on CONTENTS, not id().  `id()` is unique only among LIVE
        # objects: CPython recycles the address of a freed temporary, so two
        # different position arrays could collide and the memo then returned
        # one separation's C for another.
        #
        # A fixed-size digest, rather than the raw bytes: the digest is 16
        # bytes whatever the array's size, so there is no size cutoff, no
        # escape-hatch sentinel, and no "is this key cacheable" question to
        # get wrong -- an earlier version answered it with a membership test
        # that saw only the key's top level, so an oversized array nested in a
        # list left the sentinel buried inside and two different positions
        # shared a key.  Removing the special case removes that whole class.
        # ``tobytes()`` already serialises in C order whatever the array's
        # memory layout, so it normalises strides on its own.  Do NOT route it
        # through ``np.ascontiguousarray`` first: that downcasts an ndarray
        # SUBCLASS to a base array, discarding the subclass's own ``tobytes``
        # -- a masked array then keys on its raw buffer and two positions
        # differing only in their mask share one entry.
        cacheable = True

        def _cache_key_part(obj):
            nonlocal cacheable
            if isinstance(obj, np.ndarray):
                if obj.dtype.kind == "O":
                    # An object array's buffer is raw PyObject POINTERS, and
                    # an address is unique only among LIVE objects -- CPython
                    # recycles a freed temporary's address, so two different
                    # positions would collide.  Refuse rather than guess.
                    cacheable = False
                    return None
                return (type(obj).__name__, obj.shape, obj.dtype.str,
                        hashlib.blake2b(obj.tobytes(),
                                        digest_size=16).digest())
            if isinstance(obj, (list, tuple)):
                return tuple(_cache_key_part(v) for v in obj)
            return obj

        cache_key = (_cache_key_part(n1), t1, _cache_key_part(n2), t2)
        # ``cacheable`` is set from INSIDE the recursion, so an object array
        # nested in a list is caught too.  An earlier version tested only the
        # key's top level and let nested cases share an entry.
        if cacheable:
            hit = self._c_cache.get(cache_key)
            if hit is not None:
                return hit

        # Delegate to ``_C_value_direct`` (which adds the white-noise
        # 1-D contribution when ``model.sigma2`` is set), then store
        # in the per-arg memo.
        C_mat = self._C_value_direct(n1, t1, n2, t2)
        if cacheable:
            # Handed out read-only: the array returned IS the memo entry, so a
            # caller doing ``C += x`` would silently rewrite every later
            # lookup.  Failing loudly beats corrupting the cache.
            #
            # COPY first.  ``_C_value_direct`` ends in ``np.asarray(...)``,
            # which is the identity for a float64 array, so freezing in place
            # would flip the flag on the USER'S object -- breaking the
            # idiomatic ``c_value_fn`` that fills and returns one preallocated
            # buffer, or that returns a module-level constant correlator.
            C_mat = np.array(C_mat, copy=True)
            C_mat.flags.writeable = False
            self._c_cache[cache_key] = C_mat
            if len(self._c_cache) > _C_CACHE_MAXSIZE:
                self._c_cache.popitem(last=False)
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
        # Fast path: legacy time-only spline table -- but ONLY when there is
        # no spatial table.  `C_value` checks the spatial table FIRST, so
        # taking the legacy one here regardless meant the two accessors
        # disagreed by ~38% whenever both tables were present: this one is
        # position-blind, that one is not.  Same precedence in both now.
        if (self._c_splines is not None and self.model.diag_C
                and not _cache_has_spatial_table(self)):
            lo, hi = self._c_table_range  # type: ignore[misc]
            if lo <= t1 <= hi and lo <= t2 <= hi:
                return self._C_diagonal_from_table(t1, t2)
        C_mat = self.C_value(n, t1, n_prime, t2)
        return np.diag(C_mat)

    def __getstate__(self) -> dict:
        """Pickle without the ``C_value`` memo.

        Every parallel builder (``precompute_C_table_translation`` /
        ``_rotation`` / ``_general``) dispatches a closure that references
        ``self``, so the whole cache object is serialised into each worker
        payload.  A full memo is tens of megabytes at a modest number of
        components and grows as N^2, and a worker cannot benefit from the
        parent's entries anyway -- it recomputes what it needs.
        """
        state = self.__dict__.copy()
        state["_c_cache"] = OrderedDict()
        return state

    def clear_cache(self) -> None:
        """Clear the C value cache and spline table."""
        self._c_cache.clear()
        self._c_splines = None
        self._c_diag_splines = None
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
        # Harvest the i == j entries into a separate 1-D spline.  Zero extra
        # quadrature -- they are already in ``grids``.
        #
        # ``C(t1,t2) = int_0^{min(t1,t2)} R(t1,l) sigma2(l) R(t2,l) dl`` has a
        # derivative discontinuity of exactly ``-sigma2(t)`` on the diagonal:
        # approaching from t1 < t2 the moving upper limit contributes an extra
        # ``R(t1,t1) sigma2(t1) R(t2,t1)``, absent from the other side.  A
        # tensor-product spline is C^2 everywhere, so it smears that kink and
        # stops converging there -- measured 22.3% relative error at n_grid=41
        # and still 21.4% at n_grid=321, i.e. p = 0.009, no convergence at all,
        # while the same table is clean O(h^4) away from the diagonal.  Every
        # tadpole evaluates C(s,s), exactly on the kink.
        #
        # Along the diagonal itself ``t -> C(t,t)`` is smooth, so a 1-D cubic
        # spline restores O(h^4).
        self._c_diag_splines = [
            _diag_line_interp(ts, np.diag(grids[a])) for a in range(N)
        ]
        self._c_table_range = (t_min, t_max)

    @property
    def has_table(self) -> bool:
        """Whether a pre-computed C table is available."""
        return self._c_splines is not None

    @staticmethod
    def _on_diagonal(t1, t2):
        """Whether ``t1`` and ``t2`` are numerically the same time."""
        return np.abs(np.asarray(t1) - np.asarray(t2)) <= _DIAG_TOL * np.maximum(
            1.0, np.maximum(np.abs(np.asarray(t1)), np.abs(np.asarray(t2)))
        )

    def _C_diagonal_from_table(self, t1: float, t2: float) -> np.ndarray:
        """Look up C diagonal from spline table.

        Routes equal times through the 1-D diagonal spline: the 2-D spline
        cannot represent the kink there (see ``precompute_C_table``).
        """
        if self._c_diag_splines is not None and self._on_diagonal(t1, t2):
            t = 0.5 * (float(t1) + float(t2))
            return np.array([float(_diag_line_eval(s, t)[0])
                             for s in self._c_diag_splines])
        # ``np.squeeze`` keeps this robust across SciPy versions: newer
        # SciPy returns a 1-element 1-D array from a scalar ``grid=False``
        # call, and ``float()`` on a non-0-D array raises under NumPy >= 2.
        return np.array([
            float(np.squeeze(s(t1, t2, grid=False))) for s in self._c_splines  # type: ignore[union-attr]
        ])

    def _C_value_from_table(self, t1: float, t2: float) -> np.ndarray:
        """Look up C matrix from spline table (diagonal only).

        Shares :meth:`_C_diagonal_from_table`'s diagonal-kink routing so the
        two accessors cannot disagree about C(t, t).
        """
        N = self.model.n_components
        C_mat = np.zeros((N, N))
        diag = self._C_diagonal_from_table(t1, t2)
        for a in range(N):
            C_mat[a, a] = diag[a]
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
        if self._c_diag_splines is not None:
            # Tadpoles evaluate C(s, s): both legs are the SAME sampled time,
            # so this mask is hit exactly, not approximately.
            on_diag = self._on_diagonal(t1, t2)
            if np.any(on_diag):
                td = 0.5 * (np.asarray(t1, dtype=float)[on_diag]
                            + np.asarray(t2, dtype=float)[on_diag])
                for a, sd in enumerate(self._c_diag_splines):
                    result[on_diag, a] = _diag_line_eval(sd, td)
        return result

    def R_time_batch(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        """Evaluate R_time at arrays of time pairs (iso_R only).

        Args:
            t1: Array of shape ``(n,)`` — left times.
            t2: Array of shape ``(n,)`` — right times.

        Returns:
            Array of shape ``(n,)``.  R(t1, t2) = Θ(t1−t2) × R_time(t1, t2).
        """
        # Θ is applied here: an R propagator joining two fixed external
        # points has no integration domain to enforce it (see R_product).
        #
        # The user's R_time is called ONLY on the causal pairs, matching
        # R_product and _evaluate_r_product_general, which short-circuit
        # before calling it.  Evaluating everywhere and masking afterwards
        # would make the three sites behave differently for a model whose
        # R_time raises or overflows on acausal input — the same number,
        # but a spurious exception or RuntimeWarning through this path only.
        t1a = np.asarray(t1, dtype=float)
        t2a = np.asarray(t2, dtype=float)
        causal = t1a > t2a
        out = np.zeros(np.broadcast(t1a, t2a).shape, dtype=float)
        if causal.any():
            R_vec = np.vectorize(self.model.R_time, otypes=[float])
            b1, b2 = np.broadcast_arrays(t1a, t2a)
            out[causal] = R_vec(b1[causal], b2[causal])
        return out

    # ------------------------------------------------------------------ #
    # Quadrature-method resolution and kernel structure
    # ------------------------------------------------------------------ #

    def _direct_is_overridden(self) -> bool:
        """Whether a subclass replaced :meth:`_C_value_direct` (legacy
        extension point).  Such a cache computes C its own way: no
        quadrature is chosen for it and no quadrature kwargs are passed,
        so overrides with the original 4-argument signature keep working.
        """
        return type(self)._C_value_direct is not PropagatorCache._C_value_direct

    @property
    def c_method_resolved(self) -> str:
        """The method the next table build will use.

        ``'closed_form'`` when a C callable replaces the quadrature,
        ``'custom'`` when a subclass overrides :meth:`_C_value_direct`;
        the resolved method after a build under ``'auto'``; ``'dblquad'``
        for an unresolved ``'auto'``; else the explicit setting.
        """
        if self.c_value_fn is not None:
            return "closed_form"
        if self._direct_is_overridden():
            return "custom"
        if self.c_method != "auto":
            return self.c_method
        return self._c_method_resolved or "dblquad"

    @property
    def n_gauss_resolved(self) -> int | None:
        """Node count in force for Gauss-Legendre, else ``None``."""
        if self.c_method_resolved != "gauss_legendre":
            return None
        return self._n_gauss_resolved or self.n_gauss

    def resolve_c_method(self, t_max: float) -> tuple[str, int | None]:
        """Decide (and remember) the quadrature for a table reaching ``t_max``.

        Explicit settings are returned as they are.  ``'auto'`` runs
        :func:`select_gl_node_count` once -- later calls with a larger
        ``t_max`` re-run it, since convergence depends on the horizon.
        """
        if self.c_value_fn is not None:
            return "closed_form", None
        if self._direct_is_overridden():
            return "custom", None
        if self.c_method != "auto":
            return self.c_method, (self.n_gauss if self.c_method == "gauss_legendre" else None)
        horizon = getattr(self, "_c_method_horizon", None)
        if self._c_method_resolved is not None and horizon is not None \
                and float(t_max) <= horizon:
            return self._c_method_resolved, self._n_gauss_resolved
        n = select_gl_node_count(
            self.model, t_max, n_start=self.n_gauss,
            positions=self._probe_positions(),
        )
        if n is not None and not self._gl_is_cheaper(t_max, int(n)):
            # Reserved for a deterministic cost rule; see
            # ``_gl_is_cheaper``, which no longer races the two.
            n = None
        if n is None:
            self._c_method_resolved, self._n_gauss_resolved = "dblquad", None
        else:
            self._c_method_resolved, self._n_gauss_resolved = "gauss_legendre", int(n)
        self._c_method_horizon = float(t_max)
        return self._c_method_resolved, self._n_gauss_resolved

    def _gl_is_cheaper(self, t_max: float, n: int) -> bool:
        """``True`` when Gauss-Legendre is the rule to use, given that
        ``select_gl_node_count`` has already found ``n`` converged.

        This used to time one deep cell under each rule and take the
        winner.  Both rules are verified converged before it is called,
        so the race could never produce a wrong value -- but it made the
        CHOICE depend on machine load, and with it ``c_source`` and the
        spline table.  Two runs of the same config on a busy and an idle
        machine could resolve differently, which is a bad property for a
        package that sells reproducibility, and it made
        ``test_direct_calls_before_any_build_use_dblquad_under_auto``
        fail intermittently under load.

        Removing it costs nothing measurable.  Per ``_C_value_direct``
        call at the deep corner, on the demo1 kernel, comparing the two
        rules AT THE NODE COUNT ``auto`` RESOLVES TO (best of 5, M3
        Ultra, one core):

        ====== ========== ======== ========= ==========
        t_max  resolved n  GL       dblquad   GL wins by
        ====== ========== ======== ========= ==========
             5         20   5.7 ms    7.1 ms      1.3x
            15         20   5.7 ms   39.5 ms      6.9x
            30         30  12.5 ms   88.0 ms      7.0x
            50         30  12.5 ms  144.4 ms     11.6x
           100         45  27.9 ms  209.2 ms      7.5x
        ====== ========== ======== ========= ==========

        Gauss-Legendre wins everywhere here, because adaptive cost grows
        with the interval while a fixed-``n`` tensor rule is flat and the
        required ``n`` grows only slowly.  The one regime the race ever
        chose dblquad for is a smooth kernel over a short horizon, where
        unconditional GL costs ~1.3x.

        The decision is now a function of the inputs alone: prefer
        Gauss-Legendre whenever a converged node count exists, and fall
        back to dblquad only when none does -- which is what the caller
        already does with ``n is None``.  If the short-horizon case ever
        matters, bring it back as a THRESHOLD on the converged ``n``, not
        as a timing measurement.

        One trap if you re-derive the accuracy bound: compare the two
        rules at the RESOLVED ``n``, not at a fixed ``n = 20``.  At
        ``t_max = 50`` a fixed 20-node rule differs from dblquad by
        1.7e-04, which looks alarming -- but ``auto`` resolves to
        ``n = 30`` there, and at 30 the two agree to 2.1e-09.
        """
        return True

    def _resolve_build_method(self, c_method, n_gauss, t_max):
        """Per-build override → instance setting → ``'auto'`` resolution."""
        if c_method is None or c_method == "auto":
            method, n = self.resolve_c_method(t_max)
            if n_gauss is not None and method == "gauss_legendre":
                n = int(n_gauss)
            return method, n
        if c_method not in ("dblquad", "gauss_legendre"):
            raise ValueError(
                f"c_method must be 'auto', 'dblquad' or 'gauss_legendre'; "
                f"got {c_method!r}"
            )
        return c_method, int(self.n_gauss if n_gauss is None else n_gauss)

    @staticmethod
    def _direct_kwargs(method: str, n_gauss: int | None) -> dict:
        """Keyword arguments for ``_C_value_direct`` under ``method``.

        Forwarded only for Gauss-Legendre, so subclasses that override
        ``_C_value_direct`` with the legacy 4-argument signature keep
        working on the dblquad / closed-form paths.
        """
        if method == "gauss_legendre":
            return {"method": method, "n_gauss": int(n_gauss)}
        return {}

    def _probe_positions(self) -> tuple:
        """``(n1, n2)`` the node-count probe evaluates κ² at."""
        if self.homogeneity == "rotation":
            v = np.array([0.0, 0.0, 1.0])
            return v, v
        return np.asarray(0.0), np.asarray(0.0)

    def _c_source_label(self) -> str:
        """Short description of how C is evaluated, for progress bars."""
        m = self.c_method_resolved
        if m == "closed_form":
            return "closed form"
        if m == "custom":
            return "custom C"
        if m == "gauss_legendre":
            return f"Gauss-Legendre n={self.n_gauss_resolved}"
        return "dblquad"

    def _lazy_spatial_factor(self) -> Callable | None:
        """``r ↦ κ_x(r)`` when C factorises as ``κ_x(r) · C(0; t1, t2)``.

        True for a separable translation-invariant ``kappa2`` (the L1
        :class:`~sft_wick.workflow.specs.SeparableTranslation`) without a
        white-noise impulse -- the impulse adds an ``r``-independent term
        that would break the scaling.  A user ``c_value_fn`` disables it:
        nothing is known about that function's structure.
        """
        if self.c_value_fn is not None or self.model.sigma2 is not None \
                or self._direct_is_overridden():
            return None
        k2 = self.model.kappa2
        if getattr(k2, "separable_translation", False) and hasattr(k2, "spatial_factor"):
            return k2.spatial_factor
        return None

    def _kappa2_has_diagonal_cusp(self) -> bool:
        """Whether ``κ²(λ1, λ2)`` has a derivative jump on ``λ1 = λ2``.

        Decides whether the dblquad path splits the rectangle at the
        diagonal.  Splitting is what makes a stationary ``|λ1 − λ2|``
        kernel fast and accurate (7 ms and 1e-10 instead of 74-245 ms and
        2e-6), but on a SMOOTH kernel the three variable-limit pieces cost
        scipy 10-250× more evaluations than the one rectangle (measured
        0.2 ms → 8-50 ms), so the choice must be made per kernel.  The
        built-in kernels say so themselves (``has_diagonal_cusp``); any
        other callable is probed once: the jump ``|κ'(0⁺) − κ'(0⁻)|`` is
        estimated from one-sided differences at two step sizes and called
        a cusp when it does not shrink with the step.  A kernel that
        raises on the probe is treated as having a cusp (the safe side).
        """
        cached = getattr(self, "_cusp_cache", None)
        if cached is not None:
            return cached
        flag = getattr(self.model.kappa2, "has_diagonal_cusp", None)
        if flag is None:
            flag = _probe_diagonal_cusp(self.model, self._probe_positions())
        self._cusp_cache = bool(flag)
        return self._cusp_cache

    def _c_time_symmetric(self) -> bool:
        """Whether the diagonal C table is symmetric under ``t1 ↔ t2``.

        Requires ``κ_aa(λ1, λ2) = κ_aa(λ2, λ1)`` -- guaranteed only by the
        built-in even temporal kernels -- and diagonal C (the off-diagonal
        ``C_ab`` swaps ``R_aa ↔ R_bb`` as well).
        """
        if self.c_value_fn is not None or not self.model.diag_C \
                or self._direct_is_overridden():
            return False
        return bool(getattr(self.model.kappa2, "symmetric_in_time", False))

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
        c_method: str | None = None,
        n_gauss: int | None = None,
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
            r_max: upper bound of the r = ``|x1-x2|`` grid
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
            c_method: Quadrature method passed to
                :meth:`_C_value_direct` for every grid-point
                evaluation, in both full-grid and lazy mode.  ``None``
                (default) uses the cache instance's setting, resolving
                ``'auto'`` for this ``t_max`` (see
                :meth:`resolve_c_method`); ``'dblquad'`` /
                ``'gauss_legendre'`` override it for this build.
            n_gauss: Per-dimension GL node count for
                ``'gauss_legendre'``; ``None`` uses the instance's
                (resolved) value.

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
        method, n_gauss_val = self._resolve_build_method(c_method, n_gauss, t_max)
        direct_kwargs = self._direct_kwargs(method, n_gauss_val)

        # Lazy mode: defer per-r spline construction to first query.
        if r_max is None or n_grid_r is None:
            self._lazy_translation = _LazyTimeSplineCache(
                self, t_max, n_grid_t, mode="translation",
                n_jobs=n_jobs, direct_kwargs=direct_kwargs,
            )
            return

        # Full-grid mode: build the 3-D (t1, t2, r) spline now.
        from scipy.interpolate import RegularGridInterpolator
        from .progress import progress_map

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
                np.asarray(0.0), t1, np.asarray(r), t2, **direct_kwargs,
            )
            return i, j, k, np.array([C_mat[a, a] for a in range(N)])

        results = progress_map(
            _point, tasks, f"C table ({self._c_source_label()}, full r-grid)",
            n_jobs=n_jobs, unit="cell",
        )

        grids = [np.zeros((n_grid_t, n_grid_t, n_grid_r)) for _ in range(N)]
        for i, j, k, cvec in results:
            for a in range(N):
                grids[a][i, j, k] = cvec[a]

        self._c_translation_splines = [
            _DiagAwareGridInterp(
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
                ),
                _diag_grid_interp(
                    (ts, ts, rs), grids[a], self.interp_method,
                ),
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
        c_method: str | None = None,
        n_gauss: int | None = None,
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
            c_method: ``'dblquad'`` (default) or ``'gauss_legendre'``
                -- forwarded to :meth:`_C_value_direct` for every
                full-grid point.  See
                :meth:`precompute_C_table_translation` for details.
            n_gauss: Per-dim GL node count for
                ``c_method='gauss_legendre'``; default ``20``.

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
        method, n_gauss_val = self._resolve_build_method(c_method, n_gauss, t_max)
        direct_kwargs = self._direct_kwargs(method, n_gauss_val)

        if n_grid_cos is None:
            self._lazy_rotation = _LazyTimeSplineCache(
                self, t_max, n_grid_t, mode="rotation",
                n_jobs=n_jobs, direct_kwargs=direct_kwargs,
            )
            return

        from scipy.interpolate import RegularGridInterpolator
        from .progress import progress_map

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
            C_mat = self._C_value_direct(e1, t1, e2, t2, **direct_kwargs)
            return i, j, k, np.array([C_mat[a, a] for a in range(N)])

        results = progress_map(
            _point, tasks, f"C table ({self._c_source_label()}, full cos-grid)",
            n_jobs=n_jobs, unit="cell",
        )

        grids = [np.zeros((n_grid_t, n_grid_t, n_grid_cos)) for _ in range(N)]
        for i, j, k, cvec in results:
            for a in range(N):
                grids[a][i, j, k] = cvec[a]

        self._c_rotation_splines = [
            _DiagAwareGridInterp(
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
                ),
                _diag_grid_interp(
                    (ts, ts, coses), grids[a], self.interp_method,
                ),
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
        c_method: str | None = None,
        n_gauss: int | None = None,
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
            c_method: ``'dblquad'`` (default) or ``'gauss_legendre'``
                -- forwarded to :meth:`_C_value_direct` for every
                full-grid point.  See
                :meth:`precompute_C_table_translation` for details.
            n_gauss: Per-dim GL node count for
                ``c_method='gauss_legendre'``; default ``20``.

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
        method, n_gauss_val = self._resolve_build_method(c_method, n_gauss, t_max)
        direct_kwargs = self._direct_kwargs(method, n_gauss_val)

        if x_max is None or n_grid_x is None:
            self._lazy_general = _LazyTimeSplineCache(
                self, t_max, n_grid_t, mode="general",
                n_jobs=n_jobs, direct_kwargs=direct_kwargs,
            )
            return

        from scipy.interpolate import RegularGridInterpolator
        from .progress import progress_map

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
                np.asarray(x1), t1, np.asarray(x2), t2, **direct_kwargs,
            )
            return i, j, p, q, np.array([C_mat[a, a] for a in range(N)])

        results = progress_map(
            _point, tasks, f"C table ({self._c_source_label()}, full x-grid)",
            n_jobs=n_jobs, unit="cell",
        )

        grids = [
            np.zeros((n_grid_t, n_grid_t, n_grid_x, n_grid_x))
            for _ in range(N)
        ]
        for i, j, p, q, cvec in results:
            for a in range(N):
                grids[a][i, j, p, q] = cvec[a]

        self._c_general_interpolators = [
            _DiagAwareGridInterp(
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
                ),
                _diag_grid_interp(
                    (ts, ts, xs, xs), grids[a], self.interp_method,
                ),
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
        method: str | None = None,
        n_gauss: int | None = None,
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

        Args:
            method: ``'dblquad'`` or ``'gauss_legendre'``.  ``None``
                (default) uses :attr:`c_method`.
            n_gauss: Override per-dim node count for the Gauss-Legendre
                method.  ``None`` (default) uses :attr:`n_gauss`.
        """
        if getattr(self, "c_value_fn", None) is not None:
            # User-supplied C replaces the whole quadrature construction.
            return np.asarray(self.c_value_fn(n1, t1, n2, t2), dtype=float)

        if method is None or method == "auto":
            # Unresolved 'auto' (no table built yet, so no horizon to probe)
            # takes the robust path.
            method = self.c_method_resolved
        n_gauss_val = (
            (self._n_gauss_resolved or self.n_gauss)
            if n_gauss is None else int(n_gauss)
        )
        if method == "gauss_legendre":
            return _C_value_direct_gl(
                self.model, n1, t1, n2, t2, n_gauss=n_gauss_val,
            )
        if method != "dblquad":
            raise ValueError(
                f"unknown c_method {method!r}; expected 'dblquad' "
                f"or 'gauss_legendre'"
            )

        from scipy.integrate import dblquad, quad as _quad

        m = self.model
        N = m.n_components
        t_min = m.t_min
        C_mat = np.zeros((N, N))

        # ``R`` is a full matrix whenever the linear operator is not diagonal
        # in the chosen component basis.  Contract
        #     C_{ab} = sum_{c,d} R_{ac}(t1,l1) kappa_{cd}(l1,l2) R_{bd}(t2,l2)
        # i.e. row ``a`` of R(t1,·) against kappa against row ``b`` of R(t2,·).
        # Using only R[a,a] is correct solely for diagonal R.
        def _row(t_obs: float, lam: float, idx: int) -> np.ndarray:
            rt = m.R_time(t_obs, lam)
            if m.iso_R:
                row = np.zeros(N)
                row[idx] = float(rt)
                return row
            return np.asarray(rt, dtype=float)[idx, :]

        # Integrate the rectangle ``[t_min, t1] × [t_min, t2]`` as the three
        # cusp-free pieces used by the Gauss-Legendre path (the two
        # triangles of the square ``[t_min, min(t1,t2)]²`` on either side
        # of ``λ1 = λ2``, plus the strip outside it) rather than in one
        # go.  Stationary κ² has a ``|λ1 − λ2|`` cusp on that diagonal,
        # and an adaptive rule straddling it both stalls (per-call cost
        # rising from 74 to 245 ms with ``t`` on the demo1 kernel) and
        # reports roundoff: measured 2e-6 relative error at
        # ``epsrel=1e-10`` against the closed form, versus 1e-10 for the
        # split.  The union of the pieces is the same rectangle, so a κ²
        # without a cusp is unaffected.
        def _split_dblquad(integrand) -> float:
            if not self._kappa2_has_diagonal_cusp():
                val, _ = dblquad(integrand, t_min, t1, t_min, t2, **self.quad_opts)
                return val
            t_d = min(t1, t2)
            total = 0.0
            if t_d > t_min:
                val, _ = dblquad(integrand, t_min, t_d,
                                 lambda l1: t_min, lambda l1: l1,
                                 **self.quad_opts)
                total += val
                val, _ = dblquad(integrand, t_min, t_d,
                                 lambda l1: l1, lambda l1: t_d,
                                 **self.quad_opts)
                total += val
            if t1 > t2:
                val, _ = dblquad(integrand, t_d, t1,
                                 lambda l1: t_min, lambda l1: t_d,
                                 **self.quad_opts)
                total += val
            elif t2 > t1:
                val, _ = dblquad(integrand, t_min, t_d,
                                 lambda l1: t_d, lambda l1: t2,
                                 **self.quad_opts)
                total += val
            return total

        if m.diag_C:
            for a in range(N):
                def integrand(lam2: float, lam1: float, _a: int = a) -> float:
                    kappa_mat = np.asarray(
                        m.kappa2(n1, lam1, n2, lam2), dtype=float
                    )
                    return float(
                        _row(t1, lam1, _a) @ kappa_mat @ _row(t2, lam2, _a)
                    )
                C_mat[a, a] = _split_dblquad(integrand)
        else:
            for a in range(N):
                for b in range(N):
                    def integrand(lam2: float, lam1: float, _a: int = a, _b: int = b) -> float:
                        kappa_mat = np.asarray(
                            m.kappa2(n1, lam1, n2, lam2), dtype=float
                        )
                        return float(
                            _row(t1, lam1, _a) @ kappa_mat @ _row(t2, lam2, _b)
                        )
                    C_mat[a, b] = _split_dblquad(integrand)

        # --- White-noise (δ-correlated) component: 1-D integral ---
        if m.sigma2 is not None:
            t_upper = min(t1, t2)
            if t_upper > t_min:
                if m.diag_C:
                    for a in range(N):
                        def w_integrand(tau: float, _a: int = a) -> float:
                            sig = np.asarray(
                                m.sigma2(n1, tau, n2), dtype=float
                            )
                            return float(
                                _row(t1, tau, _a) @ sig @ _row(t2, tau, _a)
                            )
                        wval, _ = _quad(
                            w_integrand, t_min, t_upper, **self.quad_opts,
                        )
                        C_mat[a, a] += wval
                else:
                    for a in range(N):
                        for b in range(N):
                            def w_integrand(tau: float, _a: int = a, _b: int = b) -> float:
                                sig = np.asarray(
                                    m.sigma2(n1, tau, n2), dtype=float
                                )
                                return float(
                                    _row(t1, tau, _a) @ sig @ _row(t2, tau, _b)
                                )
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
        spline, calls ``self._C_value_direct`` (the user's c_fn), and
        returns either per-sample diagonals or full matrices.

        When ``model.diag_C`` is true the return shape is ``(n, N)``.
        Otherwise the full-matrix return shape is ``(n, N, N)``.

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
            if not self.model.diag_C:
                return full.astype(float, copy=False)
            # Diagonal per sample: result[i, a] = full[i, a, a].
            return np.einsum("iaa->ia", full).astype(float, copy=False)

        # Per-sample fallback. Slow but always correct.
        if self.model.diag_C:
            result = np.empty((n, N), dtype=float)
            for i in range(n):
                xi = x1_b[i]
                xj = x2_b[i]
                C_mat = np.asarray(self._C_value_direct(xi, t1[i], xj, t2[i]))
                result[i] = np.diag(C_mat)
            return result

        result_full = np.empty((n, N, N), dtype=float)
        for i in range(n):
            xi = x1_b[i]
            xj = x2_b[i]
            result_full[i] = np.asarray(self._C_value_direct(xi, t1[i], xj, t2[i]))
        return result_full

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

    #: Dynamic coupling values — mapping ``name -> callable(n_list,
    #: t_list)`` returning an ``ndarray``.
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
            if getattr(fn, "vectorized", False):
                # A callable declared under the BATCHED contract expects
                # ``(m_legs, n_samples)`` and returns
                # ``(n_samples,) + kappa_shape``.  Calling it with the
                # per-sample ``(m_legs,)`` arrays instead would hand it the
                # wrong contract silently -- some such callables broadcast and
                # return a plausible wrong shape rather than raising.  Run it
                # as a batch of one and unwrap.
                stacked = np.asarray(fn(n_list[:, None], t_list[:, None]))
                if stacked.shape[0] != 1:
                    raise ValueError(
                        f"vectorized callable {name!r} returned shape "
                        f"{stacked.shape}; expected a leading axis of "
                        f"length 1 for a single-sample evaluation."
                    )
                sample_cv[name] = stacked[0]
            else:
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
        """Vectorised batch evaluator -- returns the per-sample
        coupling as a complex array of shape ``(n_samples,) +
        prop_shape``, where ``prop_shape`` is the shape of the
        diagram's surviving :attr:`DiagramTerm.propagator_indices`
        (so ``(n_samples,)`` when the contraction is fully scalar).

        ``label_t`` and ``label_x`` map every spatial label appearing
        in any dynamic symbol's legs to a ``(n_samples,)`` time
        array / position respectively. Each ``label_x`` entry may be:

        * a scalar (the historical / 1-D translation case) — produces
          a per-leg broadcast of shape ``(n_samples,)``;
        * a ``(d,)`` vector (e.g. a 3-D unit vector under
          ``homogeneity='rotation'``) — produces a per-leg broadcast
          of shape ``(n_samples, d)``.

        The user callable then sees per-leg slices of shape ``(m,)`` or
        ``(m, d)`` respectively (or, in the vectorised contract below,
        ``(m, n_samples)`` / ``(m, n_samples, d)``). All legs of one
        symbol must have the same position shape; mixing scalar and
        vector legs raises from ``np.stack``.

        Per-symbol fast path: when the user marks a callable
        ``vectorized=True`` (e.g. via
        :class:`~sft_wick.workflow.specs.NonLocalVertex(coupling_vectorized=True)`
        or by setting ``fn.vectorized = True``), the wrapped
        callable receives ``(m_legs, n_samples)`` arrays — or
        ``(m_legs, n_samples, d)`` for d-dim positions — in a single
        call and returns a tensor of shape
        ``(n_samples,) + (N,)*order``. Otherwise we fall back to the
        ``n_samples`` per-sample calls of :meth:`evaluate_at`.

        Either way, the symbolic contraction itself runs once over
        the whole sample axis via
        :meth:`DiagramTerm.evaluate_coupling_batched`, falling back
        to a per-sample :meth:`DiagramTerm.evaluate_coupling` loop
        when the batched evaluator meets a node type it cannot
        handle.

        **Propagator-indexed output.**  When the contraction does not
        collapse to a scalar -- a κ leg index survives onto a C
        propagator, as in demo2's order-4 F³κ³ diagrams -- the
        returned array keeps those axes.  Callers must then contract
        it against the C-propagator product one index assignment at a
        time, exactly as the static branch does; see
        :meth:`DiagramIntegrand._dynamic_values`.  Before 0.4.0 this
        case raised ``NotImplementedError``; the static-vs-dynamic
        agreement that replaced it is locked by ``DC1`` in
        ``tests/test_dynamic_coupling.py``.
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
                # Positions per leg are *fixed* across samples (they
                # are the integration's external coordinates), so we
                # broadcast a leading sample axis of length n_samples.
                #
                # Two regimes:
                # * scalar position (x_val.ndim == 0): broadcast to
                #   (n_samples,) -- the historical case; produces a
                #   final n_arr of shape (m, n_samples) and a per-
                #   sample slice of shape (m,) for the user callable.
                # * d-dim vector position (x_val.ndim >= 1): broadcast
                #   to (n_samples, *x_val.shape) -- produces a final
                #   n_arr of shape (m, n_samples, *vec_shape) and a
                #   per-sample slice of shape (m, *vec_shape) for the
                #   user callable.  The user callable is responsible
                #   for handling its own per-leg position shape.
                if x_val.ndim == 0:
                    leg_x_arrs.append(np.full((n_samples,), float(x_val)))
                else:
                    leg_x_arrs.append(
                        np.broadcast_to(
                            x_val[None, ...], (n_samples,) + x_val.shape
                        )
                    )
            # Mixing scalar and vector legs (or different vector dims
            # across legs) makes ``np.stack`` raise -- a clear shape
            # error that is more useful than a silent broadcast.
            n_arr = np.stack(leg_x_arrs, axis=0)
            # scalar legs -> (m, n_samples)
            # d-dim legs  -> (m, n_samples, *vec_shape)
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

        # Probe sample 0 to learn the contracted coupling's shape --
        # ``()`` for a fully scalar contraction, or the diagram's
        # surviving propagator-index shape.  It also seeds the
        # per-sample fallback loop below.
        sample_cv0 = dict(self.static_values)
        for name, fn in self.dynamic_values.items():
            if name in per_sample_tensors:
                sample_cv0[name] = per_sample_tensors[name][0]
            else:
                n_arr, t_2d = per_symbol_legs[name]
                sample_cv0[name] = np.asarray(fn(n_arr[:, 0], t_2d[:, 0]))
        coup0 = np.asarray(
            self.diagram_term.evaluate_coupling(
                sample_cv0, self.fixed_indices,
            )
        )
        out_shape = (n_samples,) + coup0.shape

        # ------------------------------------------------------------
        # Vectorised fast path (default).
        # ------------------------------------------------------------
        # Materialise every dynamic symbol's per-sample tensor stack as
        # a single ``(n_samples, *kappa_shape)`` array, then call
        # ``DiagramTerm.evaluate_coupling_batched`` once -- replacing
        # the inner ``n_samples`` calls to ``_eval_symbolic`` with one
        # vectorised pass.
        #
        # If the symbolic ``coupling_sum`` contains a node type the
        # batched evaluator cannot handle (e.g. a ``Propagator`` or
        # ``IntegralOver`` slipped into a coupling expression), the
        # batched path raises ``NotImplementedError`` and we fall
        # back to the original per-sample loop below.  This mirrors
        # the safety net documented in
        # :func:`sft_wick.perturbation._eval_symbolic_batched`.
        try:
            batched_cv: dict = dict(self.static_values)
            for name, fn in self.dynamic_values.items():
                if name in per_sample_tensors:
                    batched_cv[name] = per_sample_tensors[name]
                else:
                    n_arr, t_2d = per_symbol_legs[name]
                    # Per-sample callable: build the (n_samples, ...)
                    # stack ourselves so the contraction is then
                    # a single ufunc pass.  This still pays one
                    # callable invocation per sample (unavoidable
                    # without ``vectorized=True``), but the symbolic
                    # contraction cost is amortised away.
                    sample0 = np.asarray(fn(n_arr[:, 0], t_2d[:, 0]))
                    stack = np.empty(
                        (n_samples,) + sample0.shape,
                        dtype=sample0.dtype,
                    )
                    stack[0] = sample0
                    for s in range(1, n_samples):
                        stack[s] = np.asarray(
                            fn(n_arr[:, s], t_2d[:, s])
                        )
                    batched_cv[name] = stack
            couplings = self.diagram_term.evaluate_coupling_batched(
                batched_cv,
                n_samples=n_samples,
                fixed_indices=self.fixed_indices,
            )
            # ``evaluate_coupling_batched`` returns shape
            # ``(n_samples,) + prop_shape``.  Cross-check it against
            # the shape the sample-0 probe produced: a mismatch means
            # the batched evaluator disagreed with the scalar one, and
            # the scalar loop below is the trustworthy route.
            if couplings.shape != out_shape:
                raise NotImplementedError(
                    "evaluate_coupling_batched returned shape "
                    f"{couplings.shape}, expected {out_shape}; "
                    "falling back to the per-sample loop."
                )
            return np.ascontiguousarray(couplings).astype(complex, copy=False)
        except NotImplementedError:
            pass  # fall through to scalar per-sample loop below

        couplings = np.empty(out_shape, dtype=complex)
        couplings[0] = coup0
        for s in range(1, n_samples):
            sample_cv = dict(self.static_values)
            for name, fn in self.dynamic_values.items():
                if name in per_sample_tensors:
                    sample_cv[name] = per_sample_tensors[name][s]
                else:
                    n_arr, t_2d = per_symbol_legs[name]
                    sample_cv[name] = np.asarray(fn(n_arr[:, s], t_2d[:, s]))
            couplings[s] = np.asarray(
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

    @property
    def _e_psi(self) -> int:
        """Number of external response legs of the observable (``E_psi``).

        ``getattr`` with a default keeps this working for
        :class:`~sft_wick.perturbation.DiagramTerm` objects deserialised from
        before the field existed.
        """
        return getattr(self.diagram_term, "n_external_response", 0)

    @property
    def observable_phase(self) -> complex:
        """``i**E_psi`` — rotates this integrand's raw value onto the reals.

        Inverse of :attr:`expected_phase`.  Used where a *batch* of complex
        values must be projected, so :func:`_real_or_raise` (scalar) does not
        apply.
        """
        return (1j) ** self._e_psi

    @property
    def expected_phase(self) -> complex:
        """``i**(-E_psi)`` — the phase a correctly-specified action produces.

        The raw value of this integrand is this phase times a real number; see
        :meth:`~sft_wick.perturbation.DiagramTerm.observable_phase_factor`.
        """
        return (-1j) ** self._e_psi

    def dynamic_coupling_array(
        self,
        times: dict[str, float],
        directions: dict[str, Any],
        default_position: Any = 0.0,
    ) -> np.ndarray:
        """Materialise this sample's coupling tensor for a callable coupling.

        The batched backends carry their own vectorised materialisation, but
        they accept only scalar isotropic R.  This is the scalar-loop
        counterpart, and it is what lets a callable coupling be combined with
        a MATRIX-valued response propagator -- the two constraints previously
        had no overlap, so that combination was computable by no backend.

        Args:
            times: ``{spatial_point: time}``.  Aliased legs of an equal-time
                non-local vertex are filled in from their representatives
                here, exactly as :meth:`evaluate` does.
            directions: ``{direction_var: value}``, keyed by
                ``spatial.direction_map`` values -- mapped back to per-label
                positions for the user callable.
            default_position: position for a coupling leg that no propagator
                attaches to, and which therefore has no ``direction_map``
                entry.
        """
        spatial = self.spatial
        if spatial.equal_time_aliases:
            times = dict(times)
            for non_rep, rep in spatial.equal_time_aliases:
                if non_rep not in times and rep in times:
                    times[non_rep] = times[rep]
        positions = {
            label: directions[dvar]
            for label, dvar in spatial.direction_map.items()
        }
        # A leg that no propagator attaches to has no ``direction_map`` entry,
        # so it carries no spatial structure and sits at the ambient position.
        # Without this the callable's own leg raises a bare ``KeyError``.
        for legs in self.dynamic_coupling.spatial_args_by_name.values():
            for label in legs:
                positions.setdefault(label, default_position)
        return self.dynamic_coupling.evaluate_at(times, positions)

    def evaluate(
        self,
        times: dict[str, float],
        directions: dict[str, Any],
        cache: PropagatorCache,
        coupling_array: np.ndarray | None = None,
    ) -> complex:
        """Evaluate the integrand at specific time + direction coordinates.

        Args:
            times: ``{spatial_point: time_value}`` for ALL spatial points.
                Aliased (non-representative) legs of equal-time non-local
                vertices may be absent --- their values are filled in from
                the canonical representatives inside this method.
            directions: ``{direction_var: value}`` for independent direction
                variables (one per R-connected component, keyed by
                the representative name from ``spatial.direction_map``).
            cache: A :class:`PropagatorCache` for propagator evaluation.
            coupling_array: this sample's coupling tensor, overriding the
                static :attr:`coupling_array`.  Required when the integrand
                carries a spacetime-dependent (callable) coupling, since the
                static attribute is then a zeros placeholder;
                :meth:`dynamic_coupling_array` produces it.

        Returns:
            Scalar value of the integrand (complex if coupling is complex).

        Raises:
            NotImplementedError: if this integrand carries a spacetime-dependent
                (callable) coupling and no ``coupling_array`` was supplied --
                it would otherwise read the zeros placeholder and return 0.
        """
        if self.dynamic_coupling is not None and coupling_array is None:
            raise NotImplementedError(
                "DiagramIntegrand.evaluate() cannot evaluate a "
                "spacetime-dependent (callable) coupling without an explicit "
                "`coupling_array`: the static attribute is a zeros "
                "placeholder on the dynamic path, so the result would "
                "silently be 0.  Pass coupling_array="
                "integrand.dynamic_coupling_array(times, directions), or use "
                "method='gauss_legendre' / method='qmc_vectorized'."
            )
        dt = self.diagram_term
        spatial = self.spatial
        coeff = (self.coupling_array if coupling_array is None
                 else coupling_array)
        model = cache.model

        # Expand the times dict to fill in aliased legs of any
        # equal_time non-local vertex; downstream propagator lookups
        # (``cache.R_product``, ``times[sp_l]`` etc.) then work without
        # special-case branches.
        if spatial.equal_time_aliases:
            times = dict(times)
            for non_rep, rep in spatial.equal_time_aliases:
                if non_rep not in times and rep in times:
                    times[non_rep] = times[rep]

        # --- R product (scalar when iso_R) ---
        #
        # ``PropagatorCache.R_product`` is intentionally scalar-only.
        # For matrix-valued R, resolve component indices alongside the
        # C contraction below so order-0 R diagrams do not try to cast an
        # (N, N) matrix to float.
        r_val = (
            cache.R_product(_kept_r_propagators(spatial), times)
            if model.iso_R else 1.0
        )

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
            if not model.iso_R:
                c_val *= self._evaluate_r_product_general(
                    times, cache, self.fixed_indices,
                )
            return complex(r_val * complex(coeff) * c_val)

        # --- Map each C propagator to its propagator-index axis ---
        idx_names = [name for name, _ in prop_idx]
        idx_name_to_axis = {name: ax for ax, name in enumerate(idx_names)}

        if model.iso_R and model.diag_C:
            return self._evaluate_diag_fast(
                r_val, times, directions, cache, idx_names, idx_name_to_axis,
                coupling_array=coeff,
            )
        else:
            return self._evaluate_general(
                r_val, times, directions, cache, idx_names, idx_name_to_axis,
                coupling_array=coeff,
            )

    def _evaluate_diag_fast(
        self,
        r_val: float,
        times: dict[str, float],
        directions: dict[str, Any],
        cache: PropagatorCache,
        idx_names: list[str],
        idx_name_to_axis: dict[str, int],
        coupling_array: np.ndarray | None = None,
    ) -> complex:
        """Fast path for iso_R + diag_C case.

        Each C propagator contributes a diagonal vector; contraction is
        element-wise multiplication then summation.
        """
        spatial = self.spatial
        coeff = (self.coupling_array if coupling_array is None
                 else coupling_array)
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
        coupling_array: np.ndarray | None = None,
    ) -> complex:
        """General path: explicit loop over propagator index combinations."""
        spatial = self.spatial
        coeff = (self.coupling_array if coupling_array is None
                 else coupling_array)
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
                c_val *= self._evaluate_r_product_general(times, cache, idx_map)

            total += c_val

        return r_val * total

    def _evaluate_r_product_general(
        self,
        times: dict[str, float],
        cache: PropagatorCache,
        idx_map: dict[str, int],
    ) -> complex:
        """Evaluate matrix-valued R propagators for one component assignment."""
        result: complex = 1.0
        for sl, sr in _kept_r_propagators(self.spatial):
            # Retarded + Itô: vanishes unless t_left > t_right.  See
            # PropagatorCache.R_product for why this is enforced here.
            if not times[sl] > times[sr]:
                return complex(0.0)
            R_mat = np.asarray(cache.R_time(times[sl], times[sr]))
            r_prop = self._find_r_propagator(sl, sr)
            if r_prop and r_prop.index_left and r_prop.index_right:
                a = self._resolve_component(r_prop.index_left, idx_map)
                b = self._resolve_component(r_prop.index_right, idx_map)
                if a is not None and b is not None:
                    result *= complex(R_mat[a, b])
                else:
                    result *= complex(np.trace(R_mat))
            else:
                result *= complex(np.trace(R_mat))
        return result

    def _find_r_propagator(self, sl: str, sr: str) -> Propagator | None:
        """Find the R propagator matching spatial_left=sl, spatial_right=sr."""
        for p in self.diagram_term.propagators:
            if p.kind == "R" and p.spatial_left == sl and p.spatial_right == sr:
                return p
        return None

    def _dynamic_values(
        self,
        couplings: np.ndarray,
        c_batches: "list[tuple[np.ndarray, str | None, str | None]]",
        r_product: np.ndarray,
        jacobians: np.ndarray | None = None,
        *,
        where: str = "",
    ) -> np.ndarray:
        """Combine a per-sample dynamic coupling with the C-propagator
        product -- the dynamic counterpart of the static branches in
        the QMC / Gauss-Legendre integrators.

        ``couplings`` comes from
        :meth:`DynamicCouplingPromise.evaluate_at_batch` and is either

        * ``(n_samples,)`` -- the contraction collapsed to a scalar, so
          the C-propagator components are fixed by
          :attr:`fixed_indices` alone and one C-product suffices; or
        * ``(n_samples,) + prop_shape`` -- a κ leg index survived onto
          a C propagator (demo2's order-4 F³κ³).  Then the C-product
          depends on the index assignment, so we sum over
          ``np.ndindex(prop_shape)`` and rebuild it per assignment,
          mirroring the static prop-indexed branch exactly.

        ``c_batches`` holds ``(C_batch, index_left, index_right)`` per
        C propagator with the batched propagator lookup already done:
        the lookup does not depend on the component assignment, so
        hoisting it keeps the per-index loop to component selection.

        Returns the ``(n_samples,)`` real integrand, multiplied by
        ``jacobians`` when given.
        """
        n_samples = int(r_product.shape[0])
        fi = self.fixed_indices

        def _c_product(idx_map: dict[str, int]) -> np.ndarray:
            cp = np.ones(n_samples)
            for C_batch, il, ir in c_batches:
                a = DiagramIntegrand._resolve_component(il, idx_map)
                b = DiagramIntegrand._resolve_component(ir, idx_map)
                cp = cp * _select_C_batch(C_batch, a, b)
            return cp

        couplings = np.asarray(couplings)
        if couplings.ndim <= 1:
            values = (
                _real_batch_or_raise(couplings, self._e_psi, where=where)
                * r_product
                * _c_product(fi)
            )
        else:
            prop_idx = self.diagram_term.propagator_indices
            idx_names = [name for name, _ in prop_idx]
            prop_shape = tuple(dim for _, dim in prop_idx)
            if couplings.shape[1:] != prop_shape:
                raise ValueError(
                    f"dynamic coupling returned per-sample shape "
                    f"{couplings.shape[1:]}, but this diagram's "
                    f"propagator indices are {prop_idx} (shape "
                    f"{prop_shape}).  A callable κ must return the "
                    f"bare tensor over its OWN legs; the contraction "
                    f"onto propagator indices is done here."
                )
            values = np.zeros(n_samples)
            for pidx in np.ndindex(*prop_shape):
                c_raw = couplings[(slice(None),) + pidx]
                # Magnitude FIRST, as in the static branch: a slice
                # that is float noise around zero carries an arbitrary
                # complex phase, and projecting it before the
                # negligibility test turns "skip this term" into a
                # hard ValueError from _real_batch_or_raise.
                if not np.any(np.abs(c_raw) >= 1e-20):
                    continue
                c_val = _real_batch_or_raise(c_raw, self._e_psi, where=where)
                idx_map = {**fi, **dict(zip(idx_names, pidx))}
                values = values + c_val * r_product * _c_product(idx_map)

        if jacobians is not None:
            values = values * jacobians
        return values

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

    def _evaluate_zero_dimensional(
        self,
        lambda_f: float,
        cache: PropagatorCache,
        *,
        direction: Any = 0,
        positions: dict[str, Any] | None = None,
        external_times: dict[str, float] | None = None,
    ) -> float:
        """Evaluate an integrand with no surviving time variables.

        Dynamic non-local couplings still need to be materialised here. This
        occurs for already-R-contracted vertices whose absorbed R legs alias
        directly onto fixed external points.
        """
        spatial = self.spatial
        if _cache_has_spatial_table(cache):
            directions = self._resolve_group_x(spatial, positions, direction)
        else:
            dir_vars = set(spatial.direction_map.values())
            directions = {d: direction for d in dir_vars}

        fixed_times, _ceiling = _resolve_external_times(
            spatial, spatial.external_points, lambda_f, external_times,
        )
        if self.dynamic_coupling is None:
            val = self.evaluate(fixed_times, directions, cache)
            return float(_real_or_raise(val, self._e_psi,
                                        where=' (zero-dimensional)'))

        n_samples = 1
        et_alias = dict(spatial.equal_time_aliases or ())

        def _times(var: str) -> np.ndarray:
            var = et_alias.get(var, var)
            return np.full(n_samples, fixed_times.get(var, lambda_f))

        r_product = np.ones(n_samples)
        for sl, sr in _kept_r_propagators(spatial):
            r_product *= cache.R_time_batch(_times(sl), _times(sr))

        spatial_aware = _cache_has_spatial_table(cache)
        group_x: dict[str, Any] | None = None
        if spatial_aware:
            group_x = self._resolve_group_x(spatial, positions, direction)

        def _lookup_C(sp_l, sp_r, t_l, t_r):
            if not spatial_aware:
                return cache.C_diagonal_batch(t_l, t_r)
            x_l = group_x[spatial.direction_map[sp_l]]  # type: ignore[index]
            x_r = group_x[spatial.direction_map[sp_r]]  # type: ignore[index]
            return cache.C_at_batch(t_l, t_r, x_l, x_r)

        c_batches = [
            (_lookup_C(sp_l, sp_r, _times(sp_l), _times(sp_r)), il, ir)
            for sp_l, sp_r, il, ir in spatial.c_propagators
        ]

        all_spatial_labels = set(spatial.direction_map.keys())
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

        couplings = self.dynamic_coupling.evaluate_at_batch(
            label_t=label_t,
            label_x=label_x,
            n_samples=n_samples,
        )
        values = self._dynamic_values(
            couplings, c_batches, r_product,
            where=' (dynamic coupling, zero-dimensional)',
        )
        return float(values[0])

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
            return _real_or_raise(result, self._e_psi,
                                  where=' (make_scipy_integrand)')

        return integrand

    def integration_bounds(
        self,
        external_times: dict[str, float],
        t_min: float = 0.0,
    ) -> list:
        """Return integration bounds for ``scipy.integrate.nquad``.

        Time ordering constraints from R causality translate to
        variable-dependent bounds.  For ``nquad``, bounds can be callables.

        The variables are ordered as ``spatial.time_integration_vars``,
        which :func:`_topological_sort_times` emits **earliest time first** —
        i.e. ``int_vars[0]`` is scipy's *innermost* integral.

        .. important::
           ``scipy.integrate.nquad`` invokes ``ranges[i]`` with the **outer**
           integration variables ``int_vars[i+1:]`` (each recursion *prepends*
           the newly bound outer variable), **not** with the inner ones.  An
           internal upper-bound source of ``int_vars[i]`` therefore always sits
           at an index ``> i`` and appears in the callback arguments at offset
           ``index(src) - i - 1``.  This is the same mapping used by
           :meth:`integrate_moment_nquad`.

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
        default_hi = max(external_times.values()) if external_times else 1.0
        # Lower limits imposed by external response legs (see
        # _causal_lower_bounds): an ordering (external, internal) is a LOWER
        # bound and cannot be expressed through ``upper_bounds``.
        lowers = _causal_lower_bounds(spatial, int_vars, external_times, t_min)

        for i, var in enumerate(int_vars):
            lo_var = lowers.get(var, t_min)
            ub_sources = upper_bounds.get(var, [])
            if not ub_sources:
                # No causal constraint — integrate up to the latest external
                # time.  Unreachable in practice: every MSR vertex carries a ψ
                # leg and is therefore the earlier endpoint of at least one R
                # ordering.
                bounds.append((lo_var, max(lo_var, default_hi)))
            else:
                def make_bound(
                    ub: list[str],
                    ext: dict[str, float],
                    ivars: list[str],
                    lo: float,
                    cur_i: int,
                    fallback: float,
                ) -> Callable:
                    def bound_func(*later_args: float) -> tuple[float, float]:
                        hi_vals: list[float] = []
                        for src in ub:
                            if src in ext:
                                hi_vals.append(ext[src])
                                continue
                            if src not in ivars:
                                hi_vals.append(fallback)
                                continue
                            j = ivars.index(src) - cur_i - 1
                            if 0 <= j < len(later_args):
                                hi_vals.append(later_args[j])
                            else:
                                hi_vals.append(fallback)
                        hi = min(hi_vals) if hi_vals else fallback
                        # An acausal / collapsed region must integrate to zero,
                        # not backwards: scipy happily returns a NEGATIVE
                        # volume for hi < lo.
                        return (lo, hi if hi > lo else lo)

                    return bound_func

                bounds.append(
                    make_bound(
                        ub_sources, external_times, int_vars, lo_var,
                        i, default_hi,
                    )
                )

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
        external_times: dict[str, float] | None = None,
    ) -> tuple[float, float]:
        """Integrate over all time variables using Quasi-Monte Carlo.

        Scalar-loop sibling of
        :meth:`integrate_moment_qmc_vectorized`; see that method's
        docstring for the ``integrate_over`` kwarg semantics.
        """
        from scipy.stats import qmc

        # A spacetime-dependent (callable) coupling used to be refused here,
        # which left the combination "callable coupling + MATRIX-valued R"
        # computable by no backend at all: this loop rejected the callable,
        # and ``qmc_vectorized`` / ``gauss_legendre`` reject matrix R.  The
        # scalar loop is in fact the natural home for the per-sample callable
        # contract -- it already visits one sample at a time -- so it now
        # materialises the coupling per sample instead.
        dyn = self.dynamic_coupling is not None

        spatial = self.spatial
        int_vars_parents_first = list(reversed(spatial.time_integration_vars))
        ext_vars = list(spatial.external_points)

        ext_int_set = _resolve_integrate_over(integrate_over, ext_vars)
        ext_integrated = [v for v in ext_vars if v in ext_int_set]
        ext_fixed = [v for v in ext_vars if v not in ext_int_set]
        fixed_times, t_ceiling = _resolve_external_times(
            spatial, ext_fixed, lambda_f, external_times, ext_integrated,
        )

        sobol_vars = ext_integrated + int_vars_parents_first
        n_ext_int = len(ext_integrated)
        n_total = len(sobol_vars)

        if _cache_has_spatial_table(cache):
            directions = self._resolve_group_x(spatial, positions, direction)
        else:
            dir_vars = set(spatial.direction_map.values())
            directions = {d: direction for d in dir_vars}

        if n_total == 0:
            ca = (self.dynamic_coupling_array(fixed_times, directions,
                                              default_position=direction)
                  if dyn else None)
            val = self.evaluate(fixed_times, directions, cache,
                                coupling_array=ca)
            return (_real_or_raise(val, self._e_psi,
                                   where=' (qmc, zero-dimensional)'), 0.0)

        parent_map: dict[str, list[str]] = defaultdict(list)
        for earlier, later in spatial.time_orderings:
            if earlier in int_vars_parents_first:
                parent_map[earlier].append(later)

        # Lower limits from external response legs (see
        # _causal_lower_bound_sources).  A *fixed* external contributes a
        # constant; a swept one contributes a per-sample value, read below
        # from ``times`` (integrated externals are drawn first).
        lowers, lower_srcs = _causal_lower_bound_sources(
            spatial, int_vars_parents_first, fixed_times, t_min,
            swept=ext_integrated,
        )

        sw_order, sw_lowers, sw_lo_c, sw_hi_c = _swept_external_order(
            spatial, ext_integrated, fixed_times, lambda_f, t_min,
        )

        # Generate Sobol samples in [0,1]^d
        sampler = qmc.Sobol(d=n_total, seed=seed)
        u_samples = sampler.random(n_samples)  # (n_samples, n_total)

        # Evaluate integrand at each sample point
        values = np.empty(n_samples)
        span = lambda_f - t_min

        for s in range(n_samples):
            u = u_samples[s]
            times: dict[str, float] = dict(fixed_times)
            jacobian = 1.0

            # Integrated externals: free in [t_min, lambda_f] except where a
            # causal ordering ties two of them together.
            for name in sw_order:
                k = ext_integrated.index(name)
                srcs = sw_lowers.get(name, ())
                lo_e = max(t_min, sw_lo_c.get(name, t_min))
                for src in srcs:
                    lo_e = max(lo_e, times[src])
                w_e = sw_hi_c.get(name, lambda_f) - lo_e
                if w_e <= 0:
                    times[name] = lo_e
                    jacobian = 0.0
                    continue
                times[name] = lo_e + u[k] * w_e
                jacobian *= w_e

            # Internal vars: bounded by causal parents
            for k, var in enumerate(int_vars_parents_first):
                idx = n_ext_int + k
                parents = parent_map.get(var, [])
                if parents:
                    hi = min(times[p] for p in parents if p in times)
                else:
                    hi = t_ceiling
                lo = lowers.get(var, t_min)
                for src in lower_srcs.get(var, ()):
                    lo = max(lo, times[src])
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

            ca = (self.dynamic_coupling_array(times, directions,
                                              default_position=direction)
                  if dyn else None)
            result = self.evaluate(times, directions, cache,
                                   coupling_array=ca)
            val = _real_or_raise(result, self._e_psi, where=' (qmc)')
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
        external_times: dict[str, float] | None = None,
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
        fixed_times, t_ceiling = _resolve_external_times(
            spatial, ext_fixed, lambda_f, external_times, ext_integrated,
        )

        # Sobol-dimensioned vars = integrated externals + internals.
        sobol_vars = ext_integrated + int_vars_pf
        n_total = len(sobol_vars)

        if n_total == 0:
            # No integration variables, but callable couplings can still be
            # dynamic after R-absorption aliases their legs to fixed externals.
            val = self._evaluate_zero_dimensional(
                lambda_f, cache, direction=direction, positions=positions,
                external_times=external_times,
            )
            return (val, 0.0)

        if spatial.r_propagators and not cache.model.iso_R:
            raise NotImplementedError(
                "method='qmc_vectorized' currently supports scalar "
                "iso_R=True response propagators only. Use method='qmc' "
                "or method='qmc_scalar' for matrix-valued R."
            )

        # Build causal parent map (per-internal-var upper bound list).
        parent_map: dict[str, list[str]] = defaultdict(list)
        for earlier, later in spatial.time_orderings:
            if earlier in int_vars_pf:
                parent_map[earlier].append(later)
        # Lower limits from external response legs (see
        # _causal_lower_bound_sources).  A *fixed* external contributes a
        # constant; a swept one contributes a per-sample column of
        # ``times_arr`` (integrated externals occupy the first n_ext_int
        # columns and are filled before this loop runs).
        lowers, lower_srcs = _causal_lower_bound_sources(
            spatial, int_vars_pf, fixed_times, t_min, swept=ext_integrated,
        )

        # Sobol samples
        sampler = qmc.Sobol(d=n_total, seed=seed)
        u = sampler.random(n_samples)
        span = lambda_f - t_min

        # Map u -> times with causal bounds (vectorized)
        times_arr = np.empty((n_samples, n_total))
        jacobians = np.ones(n_samples)

        # Integrated external vars: free in [t_min, lambda_f]
        # Swept externals: free in [t_min, lambda_f] EXCEPT where a causal
        # ordering ties two of them together (see _swept_external_order).
        # With no such edge this is bit-identical to the flat draw.
        _sw_order, _sw_lowers, _sw_lo_c, _sw_hi_c = _swept_external_order(
            spatial, ext_integrated, fixed_times, lambda_f, t_min,
        )
        for name in _sw_order:
            k = ext_integrated.index(name)
            srcs = _sw_lowers.get(name, ())
            lo_c = max(float(t_min), float(_sw_lo_c.get(name, t_min)))
            hi_c = float(_sw_hi_c.get(name, lambda_f))
            if not srcs:
                w_c = hi_c - lo_c
                times_arr[:, k] = lo_c + u[:, k] * max(w_c, 0.0)
                jacobians *= max(w_c, 0.0)
                continue
            lo_e = np.full(n_samples, lo_c)
            for src in srcs:
                lo_e = np.maximum(lo_e, times_arr[:, ext_integrated.index(src)])
            w_e = hi_c - lo_e
            ok_e = w_e > 0
            times_arr[:, k] = np.where(ok_e, lo_e + u[:, k] * w_e, lo_e)
            jacobians = np.where(ok_e, jacobians * w_e, 0.0)

        # Internal vars: bounded by parents.  Parents may be
        # (a) integrated externals — pull from times_arr; (b) fixed
        # externals — use the fixed ``lambda_f`` value; (c) other
        # internal vars — pull from times_arr.
        for k, var in enumerate(int_vars_pf):
            idx = n_ext_int + k
            parents = parent_map.get(var, [])
            if parents:
                hi = np.full(n_samples, t_ceiling)
                for p in parents:
                    if p in int_vars_pf:
                        p_idx = n_ext_int + int_vars_pf.index(p)
                        hi = np.minimum(hi, times_arr[:, p_idx])
                    elif p in ext_integrated:
                        p_idx = ext_integrated.index(p)
                        hi = np.minimum(hi, times_arr[:, p_idx])
                    else:
                        # Fixed external — use ITS OWN pinned time.
                        hi = np.minimum(hi, fixed_times.get(p, lambda_f))
            else:
                hi = np.full(n_samples, t_ceiling)

            lo_v = lowers.get(var, t_min)
            for src_v in lower_srcs.get(var, ()):
                col = ext_integrated.index(src_v)
                lo_v = np.maximum(lo_v, times_arr[:, col])
            width = hi - lo_v
            valid = width > 0
            times_arr[:, idx] = np.where(valid, lo_v + u[:, idx] * width, lo_v)
            jacobians = np.where(valid, jacobians * width, 0.0)

        # Build variable-name to column lookup.  Fixed externals are
        # NOT in times_arr; they get a special lookup that returns a
        # constant ``lambda_f`` array.
        var_to_col = {var: i for i, var in enumerate(sobol_vars)}
        fixed_t_by = {v: np.full(n_samples, t) for v, t in fixed_times.items()}
        fixed_t = np.full(n_samples, lambda_f)
        # Equal-time alias map: see analyze_spatial / SpatialStructure.
        _et_alias = dict(spatial.equal_time_aliases or ())

        def _times(var: str) -> np.ndarray:
            """Return the per-sample time array for ``var`` — pulls
            from ``times_arr`` when integrated, or the constant
            ``lambda_f`` array when fixed. Non-representative legs of
            an ``equal_time`` non-local vertex are transparently
            redirected to their canonical representative so the
            callable sees a single shared time across the m legs."""
            var = _et_alias.get(var, var)
            col = var_to_col.get(var)
            if col is not None:
                return times_arr[:, col]
            return fixed_t_by.get(var, fixed_t)

        # --- Vectorized integrand evaluation ---
        dt = self.diagram_term
        coeff = self.coupling_array
        prop_idx = dt.propagator_indices

        # R product (vectorized) — skip absorbed R's per
        # ``DiagramTerm.r_absorbed_pairs``; those factors are already
        # baked into the κ^(m)_R callable.
        r_product = np.ones(n_samples)
        for sl, sr in _kept_r_propagators(spatial):
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
            # The lookup does not depend on the component assignment,
            # so it is done once here even when the coupling is
            # prop-indexed and ``_dynamic_values`` must select
            # components per index assignment.
            c_batches = [
                (_lookup_C(sp_l, sp_r, _times(sp_l), _times(sp_r)), il, ir)
                for sp_l, sp_r, il, ir in spatial.c_propagators
            ]

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
            # ``couplings`` is (n_samples,) when the contraction is
            # scalar, or (n_samples,) + prop_shape when a κ leg index
            # survives onto a C propagator; ``_dynamic_values`` handles
            # both and does the C-component selection.
            values = self._dynamic_values(
                couplings, c_batches, r_product, jacobians,
                where=' (dynamic coupling, qmc)',
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
                    c_product *= _select_C_batch(C_diag_batch, a, b)
                else:
                    c_product *= _select_C_batch(C_diag_batch, None, None)

            values = r_product * _real_or_raise(coeff, self._e_psi, where=' (coupling)') * c_product * jacobians

        else:
            # Propagator-indexed coupling: loop over index combinations
            idx_names = [name for name, _ in prop_idx]
            prop_shape = tuple(dim for _, dim in prop_idx)
            values = np.zeros(n_samples)

            for pidx in np.ndindex(*prop_shape):
                c_raw = coeff[pidx] if coeff.ndim > 0 else coeff
                # Magnitude FIRST.  A tensor entry that is float noise around
                # zero can carry an arbitrary complex phase, and projecting it
                # before the negligibility test turns "skip this term" into a
                # hard ValueError from _real_or_raise.
                if abs(complex(c_raw)) < 1e-20:
                    continue
                c_val = _real_or_raise(c_raw, self._e_psi, where=' (coupling)')
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
                    c_prod *= _select_C_batch(C_diag_batch, a, b)

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
        external_times: dict[str, float] | None = None,
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
        fixed_times, t_ceiling = _resolve_external_times(
            spatial, ext_fixed, lambda_f, external_times, ext_integrated,
        )

        gl_vars = ext_integrated + int_vars_pf
        n_total = len(gl_vars)

        if n_total == 0:
            # No integration variables, but callable couplings can still be
            # dynamic after R-absorption aliases their legs to fixed externals.
            val = self._evaluate_zero_dimensional(
                lambda_f, cache, direction=direction, positions=positions,
                external_times=external_times,
            )
            return (val, 0.0)

        if spatial.r_propagators and not cache.model.iso_R:
            raise NotImplementedError(
                "method='gauss_legendre' currently supports scalar "
                "iso_R=True response propagators only. Use method='qmc' "
                "or method='qmc_scalar' for matrix-valued R."
            )

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
        # Lower limits from external response legs (see
        # _causal_lower_bound_sources).  A *fixed* external contributes a
        # constant; a swept one contributes a per-sample column of
        # ``times_arr`` (integrated externals occupy the first n_ext_int
        # columns and are filled before this loop runs).
        lowers, lower_srcs = _causal_lower_bound_sources(
            spatial, int_vars_pf, fixed_times, t_min, swept=ext_integrated,
        )

        span = lambda_f - t_min
        times_arr = np.empty((n_samples, n_total))
        jacobians = np.ones(n_samples)

        # Swept externals: free in [t_min, lambda_f] EXCEPT where a causal
        # ordering ties two of them together (see _swept_external_order).
        # With no such edge this is bit-identical to the flat draw.
        _sw_order, _sw_lowers, _sw_lo_c, _sw_hi_c = _swept_external_order(
            spatial, ext_integrated, fixed_times, lambda_f, t_min,
        )
        for name in _sw_order:
            k = ext_integrated.index(name)
            srcs = _sw_lowers.get(name, ())
            lo_c = max(float(t_min), float(_sw_lo_c.get(name, t_min)))
            hi_c = float(_sw_hi_c.get(name, lambda_f))
            if not srcs:
                w_c = hi_c - lo_c
                times_arr[:, k] = lo_c + u[:, k] * max(w_c, 0.0)
                jacobians *= max(w_c, 0.0)
                continue
            lo_e = np.full(n_samples, lo_c)
            for src in srcs:
                lo_e = np.maximum(lo_e, times_arr[:, ext_integrated.index(src)])
            w_e = hi_c - lo_e
            ok_e = w_e > 0
            times_arr[:, k] = np.where(ok_e, lo_e + u[:, k] * w_e, lo_e)
            jacobians = np.where(ok_e, jacobians * w_e, 0.0)

        for k, var in enumerate(int_vars_pf):
            idx = n_ext_int + k
            parents = parent_map.get(var, [])
            if parents:
                hi = np.full(n_samples, t_ceiling)
                for p in parents:
                    if p in int_vars_pf:
                        p_idx = n_ext_int + int_vars_pf.index(p)
                        hi = np.minimum(hi, times_arr[:, p_idx])
                    elif p in ext_integrated:
                        p_idx = ext_integrated.index(p)
                        hi = np.minimum(hi, times_arr[:, p_idx])
                    else:
                        # Fixed external — use ITS OWN pinned time.
                        hi = np.minimum(hi, fixed_times.get(p, lambda_f))
            else:
                hi = np.full(n_samples, t_ceiling)

            lo_v = lowers.get(var, t_min)
            for src_v in lower_srcs.get(var, ()):
                col = ext_integrated.index(src_v)
                lo_v = np.maximum(lo_v, times_arr[:, col])
            width = hi - lo_v
            valid = width > 0
            times_arr[:, idx] = np.where(valid, lo_v + u[:, idx] * width, lo_v)
            jacobians = np.where(valid, jacobians * width, 0.0)

        var_to_col = {var: i for i, var in enumerate(gl_vars)}
        fixed_t_by = {v: np.full(n_samples, t) for v, t in fixed_times.items()}
        fixed_t = np.full(n_samples, lambda_f)
        # Equal-time alias: aliased K-vertex legs share the canonical
        # representative's time variable. Resolved transparently inside
        # ``_times`` so r-propagator, c-propagator, and dynamic-coupling
        # call sites all see the collapsed time.
        _et_alias = dict(spatial.equal_time_aliases or ())

        def _times(var: str) -> np.ndarray:
            var = _et_alias.get(var, var)
            col = var_to_col.get(var)
            if col is not None:
                return times_arr[:, col]
            return fixed_t_by.get(var, fixed_t)

        # --- Vectorised integrand evaluation: identical to QMC path. ---
        dt = self.diagram_term
        coeff = self.coupling_array
        prop_idx = dt.propagator_indices

        r_product = np.ones(n_samples)
        for sl, sr in _kept_r_propagators(spatial):
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

            c_batches = [
                (_lookup_C(sp_l, sp_r, _times(sp_l), _times(sp_r)), il, ir)
                for sp_l, sp_r, il, ir in spatial.c_propagators
            ]

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
            values = self._dynamic_values(
                couplings, c_batches, r_product, jacobians,
                where=' (dynamic coupling, gauss-legendre)',
            )

        elif not prop_idx:
            c_product = np.ones(n_samples)
            for sp_l, sp_r, il, ir in spatial.c_propagators:
                t_l = _times(sp_l)
                t_r = _times(sp_r)
                C_diag_batch = _lookup_C(sp_l, sp_r, t_l, t_r)
                if il is not None and ir is not None:
                    a = DiagramIntegrand._resolve_component(il, fi)
                    b = DiagramIntegrand._resolve_component(ir, fi)
                    c_product *= _select_C_batch(C_diag_batch, a, b)
                else:
                    c_product *= _select_C_batch(C_diag_batch, None, None)
            values = r_product * _real_or_raise(coeff, self._e_psi, where=' (coupling)') * c_product * jacobians

        else:
            idx_names = [name for name, _ in prop_idx]
            prop_shape = tuple(dim for _, dim in prop_idx)
            values = np.zeros(n_samples)
            for pidx in np.ndindex(*prop_shape):
                c_raw = coeff[pidx] if coeff.ndim > 0 else coeff
                # Magnitude FIRST.  A tensor entry that is float noise around
                # zero can carry an arbitrary complex phase, and projecting it
                # before the negligibility test turns "skip this term" into a
                # hard ValueError from _real_or_raise.
                if abs(complex(c_raw)) < 1e-20:
                    continue
                c_val = _real_or_raise(c_raw, self._e_psi, where=' (coupling)')
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
                    c_prod *= _select_C_batch(C_diag_batch, a, b)
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
        external_times: dict[str, float] | None = None,
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
        # Resolved here rather than further down: the swept-external ordering
        # below needs each fixed external's own time to place constant bounds.
        fixed_times, t_ceiling = _resolve_external_times(
            spatial, ext_fixed, lambda_f, external_times, ext_integrated,
        )

        # Quadrature variables = internals + integrated externals.
        # scipy's nquad integrates index 0 innermost, and ``ranges[i]`` is
        # called with the values of ``all_vars[i+1:]`` -- so a variable can
        # only be bounded by one that sits at a HIGHER index.  A swept-to-swept
        # causal ordering therefore needs the EARLIER external placed last.
        # ``_swept_external_order`` returns earliest-first, so reverse it; with
        # no swept-to-swept edge it returns the caller's own order and this is
        # the identity.
        _sw_order, _sw_lowers, _sw_lo_c, _sw_hi_c = _swept_external_order(
            spatial, ext_integrated, fixed_times, lambda_f, t_min,
        )
        ext_ordered = list(reversed(_sw_order)) if _sw_lowers else list(
            ext_integrated)
        all_vars = int_vars + ext_ordered
        n_int = len(int_vars)
        n_total = len(all_vars)

        if _cache_has_spatial_table(cache):
            directions = self._resolve_group_x(spatial, positions, direction)
        else:
            dir_vars = set(spatial.direction_map.values())
            directions = {d: direction for d in dir_vars}

        if n_total == 0:
            val = self._evaluate_zero_dimensional(
                lambda_f, cache, direction=direction, positions=positions,
                external_times=external_times,
            )
            return (val, 0.0)

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
            return _real_or_raise(result, self._e_psi, where=' (nquad)')

        # Causal bounds for internal vars
        upper_bounds: dict[str, list[str]] = defaultdict(list)
        for earlier, later in spatial.time_orderings:
            if earlier in int_vars:
                upper_bounds[earlier].append(later)

        # Lower limits from external response legs (see
        # _causal_lower_bound_sources).  ``all_vars`` is internals-first,
        # so every swept external sits *outside* every internal variable
        # and scipy has already bound it by the time an inner range
        # callable fires -- a variable lower bound is expressible here.
        lowers, lower_srcs = _causal_lower_bound_sources(
            spatial, int_vars, fixed_times, t_min, swept=ext_integrated,
        )

        def make_bound(
            ub: list, lb: tuple, avars: list, lo: float, hi: float,
            cur_i: int,
        ) -> Callable:
            def bound_func(*later_args: float) -> tuple[float, float]:
                def outer(src: str, default: float) -> float:
                    j = avars.index(src) - cur_i - 1
                    if 0 <= j < len(later_args):
                        return later_args[j]
                    return default

                hi_v = min([hi] + [outer(s, hi) for s in ub])
                lo_v = max([lo] + [outer(s, lo) for s in lb])
                # Never integrate backwards (see integration_bounds).
                return (lo_v, hi_v if hi_v > lo_v else lo_v)
            return bound_func

        ranges: list = []
        for i, var in enumerate(all_vars):
            if i >= n_int:
                # Swept external: free in [t_min, lambda_f] unless another
                # swept external causally precedes it.
                sw_lo = _sw_lowers.get(var, ())
                lo_c = max(float(t_min), float(_sw_lo_c.get(var, t_min)))
                hi_c = float(_sw_hi_c.get(var, lambda_f))
                if sw_lo:
                    ranges.append(
                        make_bound([], tuple(sw_lo), all_vars, lo_c, hi_c, i)
                    )
                else:
                    ranges.append((lo_c, max(lo_c, hi_c)))
            else:
                lo_var = lowers.get(var, t_min)
                lo_dyn = lower_srcs.get(var, ())
                ub_sources = upper_bounds.get(var, [])
                if not ub_sources and not lo_dyn:
                    ranges.append((lo_var, max(lo_var, t_ceiling)))
                else:
                    # A fixed external never shows up in ``later_args``, so
                    # its bound must be folded into the CONSTANT part -- at
                    # ITS OWN time, not a blanket ``lambda_f``.  With every
                    # external pinned together the two coincide, which is why
                    # this stayed invisible until times could differ.
                    ub_dyn = [src for src in ub_sources
                              if src not in fixed_times]
                    hi_const = min(
                        [t_ceiling]
                        + [fixed_times[src] for src in ub_sources
                           if src in fixed_times]
                    )

                    ranges.append(
                        make_bound(ub_dyn, lo_dyn, all_vars, lo_var,
                                   hi_const, i)
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
    external_times: dict[str, float] | None = None,
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
      custom cache implements ``C_diagonal_batch`` natively) and
      ``cache.model.iso_R`` is true →
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
        if cache.model.iso_R and _cache_supports_batch_c(cache):
            return integrand.integrate_moment_qmc_vectorized(
                lambda_f, cache, t_min=t_min, direction=direction,
                n_samples=n_samples, seed=seed, positions=positions,
                integrate_over=integrate_over,
                external_times=external_times,
            )
        return integrand.integrate_moment_qmc(
            lambda_f, cache, t_min=t_min, direction=direction,
            n_samples=n_samples, seed=seed, positions=positions,
            integrate_over=integrate_over,
            external_times=external_times,
        )
    elif method == "qmc_vectorized":
        return integrand.integrate_moment_qmc_vectorized(
            lambda_f, cache, t_min=t_min, direction=direction,
            n_samples=n_samples, seed=seed, positions=positions,
            integrate_over=integrate_over,
            external_times=external_times,
        )
    elif method == "qmc_scalar":
        return integrand.integrate_moment_qmc(
            lambda_f, cache, t_min=t_min, direction=direction,
            n_samples=n_samples, seed=seed, positions=positions,
            integrate_over=integrate_over,
            external_times=external_times,
        )
    elif method == "nquad":
        return integrand.integrate_moment_nquad(
            lambda_f, cache, t_min=t_min, direction=direction,
            positions=positions,
            integrate_over=integrate_over,
            external_times=external_times,
        )
    elif method == "gauss_legendre":
        return integrand.integrate_moment_gauss_legendre(
            lambda_f, cache, t_min=t_min, direction=direction,
            n_gauss=n_gauss, positions=positions,
            integrate_over=integrate_over,
            external_times=external_times,
        )
    else:
        raise ValueError(
            f"Unknown method {method!r}; use 'qmc', 'qmc_scalar', "
            f"'qmc_vectorized', 'nquad', or 'gauss_legendre'"
        )


def _eval_single_diagram(
    dt, coupling_values, fixed_indices, lambda_f, cache, t_min, direction,
    method, n_samples, seed, positions, integrate_over, n_gauss,
    external_times=None,
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
        n_gauss=n_gauss, external_times=external_times,
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
    external_times: dict[str, float] | None = None,
    progress_tick: Callable | None = None,
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
        progress_tick: Optional ``tick(n=1)`` callable invoked after each
            diagram finishes (per batch on the parallel path).  Used by
            :meth:`~sft_wick.workflow.Expansion.sweep` to drive one bar
            across grid points × diagrams; never affects results.

    Returns:
        ``(total, details)`` where *total* is the scalar sum and
        *details* is a list of ``(estimate, error)`` per diagram.
    """
    if not diagram_terms:
        return (0.0, [])

    tick = progress_tick or (lambda n=1: None)

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
                n_gauss=n_gauss, external_times=external_times,
            )
            details.append((val, err))
            tick()
    else:
        # Parallel via joblib
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_eval_single_diagram)(
                dt, coupling_values, fixed_indices,
                lambda_f, cache, t_min, direction,
                method, n_samples, seed, positions, integrate_over,
                n_gauss, external_times,
            )
            for dt in diagram_terms
        )
        details = list(results)
        tick(len(details))

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
    external_times: dict[str, float] | None = None,
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
        t_f: External (observation) time — the DEFAULT time for every
            external point, and the ceiling for causal upper bounds.
        external_times: ``{point_name: time}`` overriding *t_f* per point,
            so ``C(x, t; y, t')`` and ``R(t, t')`` are reachable here too.
            ``None`` pins every external at *t_f*, bit-identically to
            before.  Same validation as
            :meth:`DiagramIntegrand.integrate_moment_qmc_vectorized`.
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

        # Spatial-aware dispatch, mirroring
        # ``integrate_moment_qmc_vectorized._lookup_C``.  When the cache has a
        # translation / rotation / general table, ``C_at_batch`` evaluates C at
        # the true endpoint positions and the kappa2 ratio below must NOT also
        # be applied -- that would double-count the separation.  The ratio is
        # the fallback for a cache that can only look C up by time.
        model = cache.model
        spatial_aware = _cache_has_spatial_table(cache)
        group_x = (
            DiagramIntegrand._resolve_group_x(sp, positions, 0.0)
            if spatial_aware else None
        )

        # Pre-compute spatial factors for each C propagator
        # factor = kappa2(n1, t_ref, n2, t_ref) / kappa2(0, t_ref, 0, t_ref)
        t_ref = max(t_min + 0.1, t_f * 0.5)
        kappa_00 = (
            model.kappa2(np.array([0.0]), t_ref, np.array([0.0]), t_ref)
            if not spatial_aware else None
        )  # (N, N)
        c_spatial_factors = []
        for sp_l, sp_r, _il, _ir in sp.c_propagators:
            if spatial_aware:
                c_spatial_factors.append(np.ones(model.n_components))
                continue
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

        ni = len(ivs)

        # --- Causal parent map (upper bounds) ---
        pm: dict[str, list[str]] = defaultdict(list)
        for earlier, later in sp.time_orderings:
            if earlier in ivs:
                pm[earlier].append(later)
        # --- Causal lower bounds from external response legs ---
        ext_times, t_ceiling = _resolve_external_times(
            sp, evs, t_f, external_times,
        )
        lowers = _causal_lower_bounds(sp, ivs, ext_times, t_min)

        def _lookup_C(sp_l, sp_r, t_l, t_r, ci):
            """C for propagator ``ci``, already carrying its separation.

            Spatial-aware caches resolve the separation exactly; the legacy
            time-only table cannot see position at all, so its lookup is
            scaled by the pre-computed kappa2 ratio.  Folding the ratio in
            here keeps it impossible to apply on the exact path.
            """
            if spatial_aware:
                x_l = group_x[sp.direction_map[sp_l]]
                x_r = group_x[sp.direction_map[sp_r]]
                return cache.C_at_batch(t_l, t_r, x_l, x_r)
            return (cache.C_diagonal_batch(t_l, t_r)
                    * c_spatial_factors[ci][np.newaxis, :])

        # --- Zero integration variables ---
        #
        # ``c_spatial_factors`` (the kappa2 ratio at a single ``t_ref``)
        # exists to patch ONE blind spot: the legacy time-only spline
        # table, which ignores position entirely.  ``ig.evaluate`` ->
        # ``C_value`` is position-AWARE everywhere else — through the
        # spatial fast path when a translation/rotation/general table has
        # been built, and through the ``_C_value_direct`` (dblquad /
        # ``c_value_fn``) fallback when no table has.  In those cases the
        # delegated call is *exact* and the ratio is a strictly worse
        # approximation, so applying it there would be a regression:
        # measured 4.9% at r=0.5 rising to 34.3% at r=4 for a kernel whose
        # correlation length grows with time, where delegation reproduces
        # the closed form to every printed digit.  Routing a table-less
        # cache through the batch path is worse still — ``C_diagonal_batch``
        # raises, turning a working call into a RuntimeError.
        #
        # So delegate whenever ``evaluate`` can see the positions, and use
        # the vectorised legacy-table-times-ratio path only when it cannot.
        # That is the *only* configuration in which the order-0 correlator
        # was ever wrong (it returned the coincident-point value at every
        # separation — a factor e^2 at r = 2 sigma).
        legacy_position_blind = (
            not _cache_has_spatial_table(cache)
            and getattr(cache, "_c_splines", None) is not None
            and cache.model.diag_C
        )
        if ni == 0 and not legacy_position_blind:
            et = dict(ext_times)
            total += _real_or_raise(
                ig.evaluate(et, directions, cache), ig._e_psi,
                where=" (two-point qmc, zero-dimensional)",
            )
            continue

        # --- Sobol samples (ni == 0 runs as a degenerate one-sample
        # batch with a unit Jacobian, sharing the sampled path's code so
        # the two cannot drift apart on the legacy-table convention) ---
        n_eval = n_samples if ni else 1
        t_s = np.zeros((n_eval, ni))
        jac = np.ones(n_eval)
        if ni:
            u = _qmc.Sobol(d=ni, seed=seed).random(n_samples)
            for k, var in enumerate(ivs):
                ps = pm.get(var, [])
                pi = [ivs.index(p) for p in ps if p in ivs]
                # A fixed external parent bounds from above at ITS OWN
                # time, not a blanket t_f.
                ep = [ext_times.get(p, t_f) for p in ps if p not in ivs]
                hi = (np.min(t_s[:, pi], axis=1) if pi
                      else np.full(n_samples, t_ceiling))
                if ep:
                    hi = np.minimum(hi, min(ep))
                lo_v = lowers.get(var, t_min)
                w = hi - lo_v
                ok = w > 0
                t_s[:, k] = np.where(ok, lo_v + u[:, k] * w, lo_v)
                jac = np.where(ok, jac * w, 0.0)

        # --- Full time array ---
        all_vars = evs + ivs
        var_col = {v: j for j, v in enumerate(all_vars)}
        t_arr = np.empty((n_eval, len(all_vars)))
        for j in range(len(evs)):
            t_arr[:, j] = ext_times.get(evs[j], t_f)
        for j in range(ni):
            t_arr[:, len(evs) + j] = t_s[:, j]

        # --- Vectorised R product ---
        r_prod = np.ones(n_eval)
        for sl, sr in _kept_r_propagators(sp):
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
            c_prod = np.ones(n_eval)
            for ci, (sp_l, sp_r, il, ir) in enumerate(sp.c_propagators):
                t_l = t_arr[:, var_col[sp_l]]
                t_r = t_arr[:, var_col[sp_r]]
                C_batch = _lookup_C(sp_l, sp_r, t_l, t_r, ci)
                a = DiagramIntegrand._resolve_component(il, fi)
                b = DiagramIntegrand._resolve_component(ir, fi)
                c_prod *= _select_C_batch(C_batch, a, b)
            values = r_prod * _real_or_raise(
                coeff, getattr(dt, 'n_external_response', 0),
                where=' (coupling)') * c_prod * jac

        else:
            # Propagator-indexed coupling
            idx_names = [name for name, _ in prop_idx]
            prop_shape = tuple(dim for _, dim in prop_idx)
            values = np.zeros(n_eval)
            for pidx in np.ndindex(*prop_shape):
                c_raw = coeff[pidx] if coeff.ndim > 0 else coeff
                # Magnitude FIRST.  A tensor entry that is float noise around
                # zero can carry an arbitrary complex phase, and projecting it
                # before the negligibility test turns "skip this term" into a
                # hard ValueError from _real_or_raise.
                if abs(complex(c_raw)) < 1e-20:
                    continue
                c_val = _real_or_raise(c_raw, getattr(dt, "n_external_response", 0), where=' (coupling)')
                if abs(c_val) < 1e-20:
                    continue
                idx_map = {**fi, **dict(zip(idx_names, pidx))}
                c_prod = np.ones(n_eval)
                for ci, (sp_l, sp_r, il, ir) in enumerate(
                    sp.c_propagators
                ):
                    t_l = t_arr[:, var_col[sp_l]]
                    t_r = t_arr[:, var_col[sp_r]]
                    C_batch = _lookup_C(sp_l, sp_r, t_l, t_r, ci)
                    a = DiagramIntegrand._resolve_component(il, idx_map)
                    b = DiagramIntegrand._resolve_component(ir, idx_map)
                    c_prod *= _select_C_batch(C_batch, a, b)
                values += c_val * r_prod * c_prod * jac

        values = np.where(jac > 0, values, 0.0)
        est = float(np.mean(values))
        total += est

        # Error from 8 sub-batches.  A zero-dimensional diagram is a
        # single deterministic evaluation, so it contributes no
        # quadrature variance.
        if ni:
            bs = n_samples // 8
            bm = np.array(
                [np.mean(values[j * bs : (j + 1) * bs]) for j in range(8)]
            )
            total_err_sq += (float(np.std(bm, ddof=1) / np.sqrt(8))) ** 2

    return (total, np.sqrt(total_err_sq))
