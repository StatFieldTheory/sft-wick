"""``Expansion`` — diagram-level view of the perturbative expansion.

Everything a user wants to do between ``compute_moment`` and the final
numeric result: inspect diagrams, classify by vertex composition,
draw, render LaTeX, integrate point-by-point or as a sweep.
"""

from __future__ import annotations

import itertools
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


from sft_wick.evaluate import _cache_has_spatial_table, integrate_moment
from sft_wick.expressions import Expr, Product, Symbol, Sum, Rational


def _guard_external_times(expansion, propagators, ext_times, label):
    """Fail loudly on the three ways a two-time request goes quietly wrong.

    All three were found by adversarial review of this feature, and all three
    previously produced a plausible-looking answer rather than an error.
    """

    # 1. A response leg with every external pinned together evaluates to
    #    EXACTLY 0 (Theta kills the R joining them).  Before the response field
    #    was nameable this config could not be written at all; now the natural
    #    first attempt returns a full table of zeros with no hint why.
    has_psi = any(
        name == expansion.system.field.response_name
        for (name, _comp, _sp) in (expansion.observable_repr or ())
        if isinstance(name, str)
    )
    if has_psi:
        times = list((ext_times or {}).values())
        if not ext_times or (len(set(times)) <= 1
                             and len(times) >= len(expansion.observable_repr)):
            warnings.warn(
                f"{label}: the observable carries a response leg "
                f"('{expansion.system.field.response_name}') but its external "
                f"points are all at the same time, so every value will be "
                f"exactly 0 -- the response propagator is retarded and "
                f"vanishes at equal times.  Pass "
                f"`external_times_grid={{'x': [t], 'y': [t_prime]}}` (sweep) "
                f"or `external_times=` (evaluate) to separate them.",
                UserWarning, stacklevel=3,
            )

    # 2. A time past the propagator table's horizon silently CLAMPS to the
    #    table edge, which is indistinguishable from a correct answer.
    #    ``Propagators`` exposes only build/cache/homogeneity/is_lazy, so read
    #    the horizon from the cache itself -- it is the object that clamps.
    t_range = getattr(getattr(propagators, "cache", None),
                      "_c_table_range", None)
    t_max = t_range[1] if t_range else None
    if t_max is not None and ext_times:
        over = {k: v for k, v in ext_times.items() if float(v) > float(t_max)}
        if over:
            raise ValueError(
                f"{label}: external time(s) {over} exceed the propagator "
                f"table horizon t_max={t_max}.  The C lookup would clamp to "
                f"the table edge and return a plausible but wrong value; "
                f"rebuild the propagators with a larger t_max."
            )


@dataclass(frozen=True)
class Expansion:
    """Result of :meth:`System.expand`.  Opaque to construct; use the
    accessors below.

    Attributes:
        system: the :class:`System` this expansion was built from.
        dts_by_order: ``{order: [DiagramTerm, ...]}``.
        orders: tuple of computed orders (sorted).
        observable_repr: hashable description of the observable.
        raw_result: the underlying :class:`PerturbativeResult`
            (advanced users).
    """

    system: Any  # sft_wick.workflow.System (forward decl avoids cycle)
    dts_by_order: Mapping[int, list]
    orders: tuple
    observable_repr: tuple
    raw_result: Any

    # --------------------------------------------------------------- #
    # Inspection
    # --------------------------------------------------------------- #

    def diagrams(self, order: int) -> list:
        """All :class:`DiagramTerm` objects at this order."""
        return list(self.dts_by_order[order])

    def summary(self) -> dict[int, dict[str, int]]:
        """Per-order count, plus a ``vertex_type`` histogram."""
        out: dict[int, dict[str, int]] = {}
        for o in self.orders:
            dts = self.dts_by_order[o]
            vtypes: dict[str, int] = defaultdict(int)
            ncross: dict[int, int] = defaultdict(int)
            for dt in dts:
                vtypes[self._vertex_type_label(dt)] += 1
                ncross[count_cross_group_c(dt)] += 1
            out[o] = {
                "n_diagrams": len(dts),
                "by_vertex_type": dict(vtypes),
                "by_n_cross_C": dict(ncross),
            }
        return out

    def by_vertex_type(self, order: int) -> dict[str, list]:
        """Group this order's diagrams by vertex composition label.

        Label format: the concatenation of the *unique sorted*
        coupling-symbol names appearing in the diagram's
        ``coupling_sum``.  E.g.:

        - A diagram using only the local ``F`` vertex → ``"F"``.
        - Order-2 diagram mixing local ``F`` and non-local ``K``
          (demo2 FK channel) → ``"FK"``.
        - Pure-``K`` diagram → ``"K"``.

        The label is a *set* of vertex types, not a multiset — this
        matches demo2's classification convention.
        """
        groups: dict[str, list] = defaultdict(list)
        for dt in self.dts_by_order[order]:
            groups[self._vertex_type_label(dt)].append(dt)
        return dict(groups)

    def latex(self, order: int) -> str:
        """Concatenated LaTeX of every diagram at this order."""
        parts = []
        for dt in self.dts_by_order[order]:
            parts.append(dt.to_latex())
        return r"\; + \;".join(parts) if parts else "0"

    def plot(self, order: int, i: int = 0, **kwargs):
        """Render the ``i``-th diagram at ``order`` via the package's
        :class:`DiagramRenderer`.

        Args:
            order: Perturbative order to look up in :attr:`raw_result`.
            i:     Index of the diagram within that order.
            **kwargs: Renderer-level kwargs (``figsize``, ``style``,
                ``external_label_fn``, ``vertex_label_fn``,
                ``label_format``) plus per-call ``draw`` kwargs
                (``ax``, ``title``, ``external_labels``,
                ``vertex_labels``, ``positions``, ``show_legend``).

        Returns:
            The matplotlib :class:`~matplotlib.figure.Figure`
            containing the rendered diagram.
        """
        from sft_wick import DiagramRenderer

        renderer_keys = {
            "figsize", "style", "external_label_fn",
            "vertex_label_fn", "label_format",
        }
        renderer_kwargs = {
            k: kwargs.pop(k) for k in list(kwargs) if k in renderer_keys
        }
        fd = self.raw_result.diagrams_by_order[order][i].to_feynman_diagram()
        renderer = DiagramRenderer(**renderer_kwargs)
        ax = renderer.draw(fd, **kwargs)
        return ax.figure

    # --------------------------------------------------------------- #
    # Numerical evaluation
    # --------------------------------------------------------------- #

    def evaluate(
        self,
        propagators,
        *,
        positions: dict[str, Any],
        t_final: float,
        component_pair: tuple = (0, 0),
        orders: Iterable[int] | None = None,
        vertex_types: Iterable[str] | None = None,
        integrate_over: Any = None,
        method: str = "qmc_vectorized",
        n_samples: int = 2 ** 13,
        seed: int | None = 42,
        n_jobs: int = 1,
        n_gauss: int = 8,
        external_times: dict[str, float] | None = None,
        _progress_tick: Any = None,
    ):
        """Integrate the expansion at a single ``(positions, t_final,
        component_pair)`` point.  Returns a :class:`Result`.

        Args:
            propagators: :class:`Propagators` from
                :meth:`System.propagators`.
            positions: ``{spatial_arg: x_value}`` mapping — e.g.
                ``{"x": 0.0, "y": 0.5}``.
            t_final: upper time bound for external-time integration
                (``lambda_f``).
            component_pair: ``(a, b)`` component indices for the
                observable (e.g. ``(1, 1)``).  For scalar
                observables use ``(0, 0)``.
            orders: subset of the expansion's orders; ``None`` uses
                all.
            vertex_types: subset of the vertex-composition labels
                (e.g. ``{"F"}``, ``{"FK"}``) to include — labels
                match :meth:`by_vertex_type` keys.  ``None`` ⇒ all.
                Useful for computing a single channel, or for
                skipping channels that require a bespoke
                integrator (e.g. non-local K whose coupling is
                spacetime-dependent; see demo2).
            integrate_over: Controls which **external** points have
                their time integrated.

                - ``None`` (default — **physics observable**): all
                  externals held fixed at ``t_final``.  Matches the
                  equal-time correlator ``⟨φ(t_f) · φ(t_f)⟩`` that
                  is compared to MC data and demo notebooks.
                - ``"all"``: all externals integrated over
                  ``[t_min, t_final]`` — the time-integrated moment
                  ``⟨∫φ(t)dt · ∫φ(t')dt'⟩``.  Natural e.g. for
                  weak-lensing line-of-sight integrals.
                - Iterable of external-point names: mixed — those
                  listed are integrated, others fixed.  E.g.
                  ``{"x"}`` for a source integrated along the line
                  of sight × a detector field at ``t_final``.

            method: time-integrator selector. Recommended choice depends on
                the diagram's number of internal time-integration variables
                ``d = len(time_integration_vars)`` and integrand smoothness:

                .. list-table::
                   :header-rows: 1
                   :widths: 28 38 34

                   * - Method
                     - Best for
                     - Trade-off
                   * - ``'qmc_vectorized'`` (default)
                     - ``d >= 6`` / non-smooth integrand
                     - ``~ 1/sqrt(n_samples)`` bias
                   * - ``'gauss_legendre'``
                     - ``d <= 5`` smooth (the typical sft-wick case)
                     - exponential convergence in ``n_gauss``; cost ``n_gauss^d``
                   * - ``'nquad'``
                     - Adaptive 1-3D fallback
                     - slow; raises ``NotImplementedError`` on dynamic-coupling
                   * - ``'qmc'`` / ``'qmc_scalar'``
                     - Compatibility / debugging
                     - slow Python loop

                See :doc:`/user_guide/workflow` "Choosing an integrator"
                for the full decision matrix and worked examples.

            n_samples, seed: forwarded to the integrator (QMC only).
            n_gauss: nodes per dimension for ``method='gauss_legendre'``
                (default 8 — exact for polynomials up to degree 15).
                Cost scales as ``n_gauss^d``; bump to 12-20 for
                stiff integrands at large ``t_final``.
        """
        from .result import Result

        orders_list = (
            sorted(set(int(o) for o in orders))
            if orders is not None
            else list(self.orders)
        )
        vtype_filter = (
            None if vertex_types is None else set(vertex_types)
        )
        coupling_values = self.system.build_coupling_values()
        fi = _component_indices(component_pair, self.observable_repr)

        # Collect tasks (diagram_term + metadata) up-front, in stable order,
        # then dispatch the whole batch to ``integrate_diagrams`` which handles
        # the sequential vs joblib loky path internally (n_jobs=1 stays serial,
        # bit-identical to the pre-refactor loop).
        tasks: list[tuple[int, int, str, Any]] = []
        for order in orders_list:
            for i, dt in enumerate(self.dts_by_order[order]):
                vtype = self._vertex_type_label(dt)
                if vtype_filter is not None and vtype not in vtype_filter:
                    continue
                tasks.append((order, i, vtype, dt))

        diagram_terms = [task[3] for task in tasks]

        from sft_wick.evaluate import integrate_diagrams

        # A cache built outside `System.propagators()` (see
        # `propagators_from_cache`) carries no record of which system it was
        # meant for.  An UNDER-counted one silently returns a wrong number
        # rather than failing, so check here, where both counts are known.
        _cache_n = getattr(getattr(propagators, "cache", None), "model", None)
        _cache_n = getattr(_cache_n, "n_components", None)
        if _cache_n is not None and int(_cache_n) != int(self.system.n_components):
            raise ValueError(
                f"the propagator cache has n_components={_cache_n} but the "
                f"expansion's system has {self.system.n_components}.  A "
                f"mismatched cache does not fail on its own -- it silently "
                f"returns a wrong number."
            )

        _guard_external_times(
            self, propagators, external_times, "Expansion.evaluate",
        )

        _total, details = integrate_diagrams(
            diagram_terms,
            coupling_values=coupling_values,
            lambda_f=t_final,
            cache=propagators.cache,
            method=method,
            n_samples=n_samples,
            seed=seed,
            fixed_indices=fi,
            n_jobs=n_jobs,
            positions=positions,
            integrate_over=integrate_over,
            n_gauss=n_gauss,
            external_times=external_times,
            progress_tick=_progress_tick,
        )

        per_diagram = []
        per_order: dict[int, float] = defaultdict(float)
        per_vtype: dict[str, float] = defaultdict(float)
        total = 0.0
        for (order, i, vtype, dt), (val, err) in zip(tasks, details):
            per_diagram.append({
                "order": order,
                "diagram_idx": i,
                "vertex_type": vtype,
                "n_cross_C": count_cross_group_c(dt),
                "value": val,
                "error": err,
            })
            per_order[order] += val
            per_vtype[vtype] += val
            total += val

        return Result(
            total=total,
            by_order=dict(per_order),
            by_vertex_type=dict(per_vtype),
            per_diagram=per_diagram,
            positions=dict(positions),
            t_final=t_final,
            component_pair=tuple(component_pair),
            n_samples=n_samples,
            seed=seed,
        )

    def sweep(
        self,
        propagators,
        *,
        positions_grid: dict[str, list],
        t_final_grid: list,
        external_times_grid: dict[str, list] | None = None,
        component_pairs: Iterable[tuple] = ((0, 0),),
        orders: Iterable[int] | None = None,
        vertex_types: Iterable[str] | None = None,
        integrate_over: Any = None,
        method: str = "qmc_vectorized",
        n_samples: int = 2 ** 13,
        seed: int | None = 42,
        n_jobs: int = 1,
        evaluate_n_jobs: int = 1,
        n_gauss: int = 8,
        progress: Any = None,
    ):
        """Cartesian-product sweep over positions, t_final, and
        component pairs.

        Args:
            positions_grid: ``{spatial_arg: [list of values]}``.
                Each key's list is swept independently; result is the
                full Cartesian product.  E.g.
                ``{"x": [0.0], "y": [0.0, 0.5, 1.0]}``.
            t_final_grid: list of upper time bounds.  Also the default time
                for any external point not named in ``external_times_grid``.
            external_times_grid: ``{point: [times]}`` pinning external points
                at UNEQUAL times, swept as a further Cartesian axis (same
                shape as ``positions_grid``).  Required for two-time
                observables: with every external at one time, Θ kills the R
                joining them and any observable carrying a response leg is
                identically 0.  Each named point adds a ``t_<point>`` column
                to the result rows.  Omit to pin everything at ``t_final``,
                which reproduces the pre-existing rows exactly.
            component_pairs: list of ``(a, b)`` component index
                tuples.
            vertex_types: optional filter — same semantics as in
                :meth:`evaluate`; only diagrams whose
                :meth:`_vertex_type_label` lies in this set are
                integrated.  ``None`` ⇒ all channels.
            n_jobs: parallelise over Cartesian-product grid points
                (``positions × t_final × component_pairs``).
                ``1`` (default) preserves the original sequential
                behaviour — bit-identical when seed is fixed.
                ``-1`` uses all CPU cores via joblib loky.
            evaluate_n_jobs: parallelise over diagrams **inside** each
                grid point's :meth:`evaluate` call.  Mutually
                exclusive with ``n_jobs > 1`` (nested loky pools are
                not supported); the dispatcher raises if both are
                set.  Use ``n_jobs > 1`` when the sweep grid is
                large; use ``evaluate_n_jobs > 1`` when each grid
                point has many diagrams (typical at orders >= 2).
            method, n_samples, seed, n_gauss: integrator knobs --
                see :meth:`evaluate` for the recommendation matrix
                and :doc:`/user_guide/workflow` "Choosing an
                integrator".  ``'gauss_legendre'`` with ``n_gauss=8``
                is the right default for ``d ≤ 5`` smooth integrands
                (exponential convergence, deterministic, no seed).
            progress: progress-bar setting (``True`` / ``False`` /
                ``(desc, done, total)`` callable / ``None`` = inherit
                from the environment).  One bar spans every grid
                point × diagram; it advances per diagram on the serial
                path and per grid point under ``n_jobs > 1``.  See
                :mod:`sft_wick.progress`.  Never affects results.

        Returns:
            :class:`SweepResult` with a pandas-friendly tidy table.
        """
        from sft_wick.progress import progress as _progress_scope

        with _progress_scope(progress):
            return self._sweep(
                propagators,
                positions_grid=positions_grid,
                t_final_grid=t_final_grid,
                external_times_grid=external_times_grid,
                component_pairs=component_pairs,
                orders=orders,
                vertex_types=vertex_types,
                integrate_over=integrate_over,
                method=method,
                n_samples=n_samples,
                seed=seed,
                n_jobs=n_jobs,
                evaluate_n_jobs=evaluate_n_jobs,
                n_gauss=n_gauss,
            )

    def _sweep(
        self,
        propagators,
        *,
        positions_grid,
        t_final_grid,
        external_times_grid,
        component_pairs,
        orders,
        vertex_types,
        integrate_over,
        method,
        n_samples,
        seed,
        n_jobs,
        evaluate_n_jobs,
        n_gauss,
    ):
        from .result import SweepResult

        if int(n_jobs) != 1 and int(evaluate_n_jobs) != 1:
            raise ValueError(
                "Specify exactly one of {n_jobs, evaluate_n_jobs} > 1; "
                "nested joblib loky pools are not supported."
            )

        orders_list = (
            sorted(set(int(o) for o in orders))
            if orders is not None
            else list(self.orders)
        )

        pos_keys = list(positions_grid.keys())
        pos_values = [positions_grid[k] for k in pos_keys]

        # ``external_times_grid`` mirrors ``positions_grid``: one list per
        # external point, swept as a further Cartesian axis.  This is what
        # makes two-time observables -- R(t, t') and C(t, t'), the DMFT order
        # parameters -- reachable declaratively; with every external pinned at
        # a single ``t_final``, Theta kills the R joining them and any
        # observable carrying an external response leg is identically 0.
        et_keys = list(external_times_grid.keys()) if external_times_grid else []
        et_values = [external_times_grid[k] for k in et_keys]
        if any(not v for v in et_values):
            raise ValueError(
                f"external_times_grid has an empty list for "
                f"{[k for k, v in zip(et_keys, et_values) if not v]}; the "
                f"Cartesian product would be empty and the sweep would return "
                f"no rows."
            )
        # The row dict is flat, so a `t_<point>` column that coincides with an
        # existing key would silently OVERWRITE it -- e.g. an external named
        # `final` shadowing the sweep's own `t_final`, or a spatial point
        # literally named `t_x`.
        _reserved = set(pos_keys) | {
            "t_final", "a", "b", "order", "diagram_idx", "vertex_type",
            "n_cross_C", "value", "error",
        }
        _clash = {f"t_{k}" for k in et_keys} & _reserved
        if _clash:
            raise ValueError(
                f"external_times_grid would emit column(s) {sorted(_clash)}, "
                f"which collide with existing sweep columns and would "
                f"silently overwrite them.  Rename the affected point(s)."
            )

        # Flatten the Cartesian product to a list of grid-point tasks.
        grid_tasks: list[tuple[dict, Any, dict, tuple]] = []
        for pos_tuple in itertools.product(*pos_values):
            positions = dict(zip(pos_keys, pos_tuple))
            for t_f in t_final_grid:
                for et_tuple in (itertools.product(*et_values)
                                 if et_keys else [()]):
                    ext_times = dict(zip(et_keys, et_tuple)) or None
                    for (a, b) in component_pairs:
                        grid_tasks.append((positions, t_f, ext_times, (a, b)))

        # Warn once per distinct external-times combination rather than once
        # per grid point, which would bury the message.
        for _et in ({tuple(sorted((ext or {}).items()))
                     for _p, _t, ext, _c in grid_tasks}):
            _guard_external_times(
                self, propagators, dict(_et) or None, "Expansion.sweep",
            )

        # A spatially structureless cache returns the same C at every
        # separation, so a positions sweep yields a column of identical
        # numbers beside a varying position column.  That is correct for a
        # disorder-averaged single-site cache such as ``spectral_cache`` and a
        # silent mistake for anyone who expected a separation dependence.
        #
        # The question is "does this SWEEP cover more than one distinct
        # configuration of external positions" -- which is knowable only here,
        # from the grid.  Asking it per grid point instead (does this one
        # point put its two externals at different places?) gets both halves
        # wrong: it says nothing about a sweep over a single position key, and
        # it fires on every ordinary single-separation ``evaluate(x != y)``.
        _configs = {tuple(sorted(
            (k, tuple(np.atleast_1d(v).ravel().tolist()))
            for k, v in _p.items()
        )) for _p, _t, _e, _c in grid_tasks}
        if len(_configs) > 1 and not _cache_has_spatial_table(
                getattr(propagators, "cache", None)):
            warnings.warn(
                f"Expansion.sweep: the propagator cache has no spatial "
                f"structure, so C is the same at every separation -- the "
                f"{len(_configs)} position configurations in this sweep will "
                f"all give identical values.  That is expected for a "
                f"disorder-averaged (single-site) cache such as "
                f"`spectral_cache`; it is a mistake if you expected a "
                f"separation dependence.",
                UserWarning, stacklevel=2,
            )

        # Diagrams per grid point, for the progress bar's total.
        vtype_filter = None if vertex_types is None else set(vertex_types)
        n_diag = sum(
            1 for o in orders_list for dt in self.dts_by_order[o]
            if vtype_filter is None or self._vertex_type_label(dt) in vtype_filter
        )

        def _eval_grid_point(task, tick=None):
            positions, t_f, ext_times, comp = task
            res = self.evaluate(
                propagators,
                positions=positions,
                t_final=t_f,
                external_times=ext_times,
                component_pair=comp,
                orders=orders_list,
                vertex_types=vertex_types,
                integrate_over=integrate_over,
                method=method,
                n_samples=n_samples,
                seed=seed,
                n_jobs=evaluate_n_jobs,
                n_gauss=n_gauss,
                _progress_tick=tick,
            )
            return positions, t_f, ext_times, comp, res

        from sft_wick.progress import progress_bar

        desc = f"sweep ({len(grid_tasks)} grid points x {n_diag} diagrams)"
        with progress_bar(len(grid_tasks) * n_diag, desc, unit="diagram") as tick:
            if int(n_jobs) == 1 or len(grid_tasks) <= 2:
                # Sequential — bit-identical to the pre-refactor nested loops.
                results = [_eval_grid_point(t, tick) for t in grid_tasks]
            else:
                from joblib import Parallel, delayed
                gen = Parallel(n_jobs=n_jobs, backend="loky",
                               return_as="generator")(
                    delayed(_eval_grid_point)(t) for t in grid_tasks
                )
                results = []
                for r in gen:
                    results.append(r)
                    tick(n_diag)

        rows = []
        for positions, t_f, ext_times, (a, b), res in results:
            # Hashable normalisation: d-dim vector positions arrive as
            # ``list`` or ``np.ndarray``; pandas ``groupby`` (used in
            # :meth:`SweepResult.totals`) factorises group keys via a
            # hash table, which rejects list-typed cells with
            # ``TypeError: unhashable type: 'list'``.  Coerce here so
            # downstream aggregation works for both scalar and d-dim
            # positions.
            hashable_positions = {
                k: (tuple(v.tolist()) if hasattr(v, "tolist")
                    else (tuple(v) if isinstance(v, (list, tuple))
                          else v))
                for k, v in positions.items()
            }
            for pd_row in res.per_diagram:
                rows.append({
                    **hashable_positions,
                    "t_final": t_f,
                    # ``t_<point>`` rather than the bare name, which is
                    # already taken by that point's spatial position.
                    **{f"t_{k}": (ext_times or {}).get(k) for k in et_keys},
                    "a": a, "b": b,
                    **pd_row,
                })
        return SweepResult(
            rows=rows, position_keys=tuple(pos_keys),
            external_time_keys=tuple(f"t_{k}" for k in et_keys),
        )

    # --------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------- #

    @staticmethod
    def _vertex_type_label(dt) -> str:
        """Extract the sorted unique coupling-symbol names from
        ``dt.coupling_sum`` and join them into a single string
        (matches demo2's convention: ``{'F'}`` → ``"F"``,
        ``{'F', 'K'}`` → ``"FK"``)."""
        names = _collect_symbol_names(dt.coupling_sum)
        if not names:
            return ""   # order-0 trivial
        return "".join(sorted(names))


# =========================================================================
# Module-level helpers
# =========================================================================


def count_cross_group_c(dt) -> int:
    """Number of C propagators whose endpoints land in distinct
    direction groups (same helper as
    :class:`tests.test_deductive_numerics.TestSpatialAwareCache` and
    :mod:`examples.demo1.validate_phase5`)."""
    spatial = dt.analyze_spatial()
    n = 0
    for p in dt.propagators:
        if p.kind != "C":
            continue
        d_l = spatial.direction_map[p.spatial_left]
        d_r = spatial.direction_map[p.spatial_right]
        if d_l != d_r:
            n += 1
    return n


def _collect_symbol_names(expr: Expr) -> set[str]:
    """Walk a coupling-sum expression tree and return the set of
    :class:`Symbol` names referenced (the coupling-tensor names)."""
    out: set[str] = set()

    def walk(e: Expr):
        if isinstance(e, Symbol):
            out.add(e.name)
            return
        if isinstance(e, Rational):
            return
        if isinstance(e, (Product, Sum)):
            for f in e.factors if isinstance(e, Product) else e.terms:
                walk(f)
            return
        # Catch other Expr subclasses that may wrap children
        for attr in ("expr", "body", "integrand"):
            child = getattr(e, attr, None)
            if isinstance(child, Expr):
                walk(child)

    walk(expr)
    return out


def _component_indices(component_pair, observable_repr):
    """Build the ``fixed_indices`` dict for
    :meth:`DiagramTerm.build_integrand` from the observable's
    component-index names.

    For observable ``(phi_a(x), phi_b(y))`` this yields
    ``{"a": component_pair[0], "b": component_pair[1]}``.  For a
    scalar observable (no component indices), returns ``{}``.
    """
    fi: dict[str, int] = {}
    for i, (name, comp, spatial) in enumerate(observable_repr):
        if comp is not None and i < len(component_pair):
            fi[comp] = int(component_pair[i])
    return fi
