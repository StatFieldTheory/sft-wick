"""``Expansion``: diagram-level view of the perturbative expansion.

Covers the steps between ``compute_moment`` and the final numeric
result: inspect diagrams, classify by vertex composition, draw, render
LaTeX, integrate point-by-point or as a sweep.
"""

from __future__ import annotations

import collections.abc
import itertools
import numbers
import string
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


from sft_wick._dense_contraction import DEFAULT_CONTRACTION_MAX_BYTES
from sft_wick.evaluate import DEFAULT_GL_CHUNK_SIZE
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
        if not ext_times or (
            len(set(times)) <= 1 and len(times) >= len(expansion.observable_repr)
        ):
            warnings.warn(
                f"{label}: the observable carries a response leg "
                f"('{expansion.system.field.response_name}') but its external "
                f"points are all at the same time, so every value will be "
                f"exactly 0 -- the response propagator is retarded and "
                f"vanishes at equal times.  Pass "
                f"`external_times_grid={{'x': [t], 'y': [t_prime]}}` (sweep) "
                f"or `external_times=` (evaluate) to separate them.",
                UserWarning,
                stacklevel=3,
            )

    # 2. A time past the propagator table's horizon silently CLAMPS to the
    #    table edge, which is indistinguishable from a correct answer.
    #    ``Propagators`` exposes only build/cache/homogeneity/is_lazy, so read
    #    the horizon from the cache itself, which is the object that clamps.
    t_range = getattr(getattr(propagators, "cache", None), "_c_table_range", None)
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


def _guard_single_site(expansion, diagram_terms, positions) -> None:
    """Refuse external points at different positions that a diagram of a
    multiplicative-noise system joins through response propagators.

    R carries ``δ(n − n')``, so the points of one R-connected group share a
    position.  A group holds two external points only through a vertex with
    two ψ legs (or an external ψ), and a :class:`MultiplicativeImpulse`
    brings such vertices.  They are local: they describe one SDE per point,
    and say nothing about how the noise at two points is correlated.

    ``evaluate.py::_require_one_position_per_group`` refuses the same
    spelling for any system at the L0 entry points, so this is not the only
    thing standing between the user and the arbitrary pick; it fires first
    and says what the local vertices do not describe, which the general
    message cannot.
    """
    from .specs import MultiplicativeImpulse

    if not positions or not isinstance(
        expansion.system.noise.sigma2, MultiplicativeImpulse
    ):
        return
    values = [np.asarray(v) for v in positions.values()]
    if all(np.array_equal(v, values[0]) for v in values[1:]):
        return
    for dt in diagram_terms:
        spatial = dt.analyze_spatial()
        external = set(spatial.external_points)
        for group in spatial.direction_groups:
            pts = sorted(p for p in group if p in external and p in positions)
            for p in pts[1:]:
                if not np.array_equal(
                    np.asarray(positions[p]), np.asarray(positions[pts[0]])
                ):
                    raise ValueError(
                        f"Expansion.evaluate: the external points "
                        f"'{pts[0]}' and '{p}' are at different positions "
                        f"({positions[pts[0]]!r} and {positions[p]!r}), and "
                        f"a chain of response propagators joins them in a "
                        f"diagram of this system.  A MultiplicativeImpulse "
                        f"lowers to local vertices, so it describes one SDE "
                        f"per spatial point; the value at two positions "
                        f"would depend on how the noise at different points "
                        f"is correlated, which those vertices do not "
                        f"describe.  Put the external points at one position "
                        f"(distinct labels and distinct external_times are "
                        f"fine)."
                    )


@dataclass(frozen=True)
class _PreparedSweepPoint:
    """Static bindings for one component tuple, owned by one sweep call."""

    tasks: tuple
    coupling_values: dict
    integrands: tuple
    cross_counts: tuple


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
        target: ``'moment'`` or ``'connected'`` (see :meth:`System.expand`).
    """

    system: Any  # sft_wick.workflow.System (forward decl avoids cycle)
    dts_by_order: Mapping[int, list]
    orders: tuple
    observable_repr: tuple
    raw_result: Any
    # Default also applies when loading pre-target pickled expansions.
    target: str = "moment"

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

        The label is a *set* of vertex types, not a multiset, matching
        demo2's classification convention.
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
            "figsize",
            "style",
            "external_label_fn",
            "vertex_label_fn",
            "label_format",
        }
        renderer_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in renderer_keys}
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
        component_pair: tuple | None = None,
        orders: int | Iterable[int] | None = None,
        vertex_types: str | Iterable[str] | None = None,
        integrate_over: Any = None,
        method: str = "qmc_vectorized",
        n_samples: int = 2**13,
        seed: int | None = 42,
        n_jobs: int = 1,
        n_gauss: int = 8,
        external_times: dict[str, float] | None = None,
        _progress_tick: Any = None,
        _prepared: _PreparedSweepPoint | None = None,
        gl_chunk_size: int | None = DEFAULT_GL_CHUNK_SIZE,
        qmc_coupling_chunk_size: int | None = None,
        propagator_cache_bytes: int | None = None,
        propagator_cache_contract: str | None = None,
        contraction_backend: str = "ordered",
        contraction_max_bytes: int = DEFAULT_CONTRACTION_MAX_BYTES,
        _lookup_cache: Any = None,
    ):
        """Integrate the expansion at a single ``(positions, t_final,
        component_pair)`` point.  Returns a :class:`Result`.

        Args:
            propagators: :class:`Propagators` from
                :meth:`System.propagators`.
            positions: ``{spatial_arg: x_value}`` mapping, e.g.
                ``{"x": 0.0, "y": 0.5}``.
            t_final: upper time bound for external-time integration
                (``lambda_f``).
            component_pair: component indices of the observable, one
                per operator in order: ``(a, b)`` for
                ``<phi_a(x) phi_b(y)>``, ``(a, b, c)`` for a 3-point
                observable.  ``None`` (default) is ``(0, ..., 0)``.  A
                tuple of the wrong length, or with an index outside
                ``0..N-1``, raises ``ValueError``, and so does a tuple
                that gives two operators sharing one component letter
                (``<phi_a(x) phi_a(y)>``) two different indices.
            orders: subset of the expansion's orders, as one integer or a
                list of them; ``None`` uses all.  An order the expansion
                does not hold, an empty list, and a boolean, string or
                float raise.  A selected order at which the expansion holds
                no diagram is listed in ``by_order`` as 0.0.
            vertex_types: the vertex-composition labels to include, as one
                string (``"FK"``) or any other iterable of strings (a
                list, a set, a NumPy array); labels match
                :meth:`by_vertex_type` keys.  ``None`` ⇒ all.  A label no
                diagram at the selected orders carries raises
                ``ValueError`` naming the labels present, so a misspelt
                label cannot drop its channel.  Under the connected target
                a label such a diagram could carry (sorted names of at most
                that many of the system's vertices) is accepted instead,
                and its coefficient is the computed zero.  Useful for
                computing a single channel, or for
                skipping channels that require a bespoke
                integrator (e.g. non-local K whose coupling is
                spacetime-dependent; see demo2).
            integrate_over: Controls which **external** points have
                their time integrated.

                - ``None`` (default, the **physics observable**): all
                  externals held fixed at ``t_final``.  Matches the
                  equal-time correlator ``⟨φ(t_f) · φ(t_f)⟩`` that
                  is compared to MC data and demo notebooks.
                - ``"all"``: all externals integrated over
                  ``[t_min, t_final]``, the time-integrated moment
                  ``⟨∫φ(t)dt · ∫φ(t')dt'⟩``.  Natural e.g. for
                  weak-lensing line-of-sight integrals.
                - Iterable of external-point names: mixed, where those
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
                     - ``d >= 6`` / kinks inside coupling callables
                     - ``~ 1/sqrt(n_samples)`` bias
                   * - ``'gauss_legendre'``
                     - ``d <= 5`` smooth (the typical sft-wick case), or
                       kinked by white noise or a vertex with several ψ
                       legs (split out)
                     - exponential convergence in ``n_gauss``; cost
                       ``n_gauss^d`` per consistent order of kinked pairs
                   * - ``'nquad'``
                     - Adaptive 1-3D fallback
                     - slow; a callable coupling costs one call per point
                   * - ``'qmc'`` / ``'qmc_scalar'``
                     - Compatibility / debugging
                     - slow Python loop

                See :doc:`/user_guide/workflow` "Choosing an integrator"
                for the full decision matrix and worked examples.

            n_samples, seed: forwarded to the integrator (QMC only).
            n_gauss: nodes per dimension for ``method='gauss_legendre'``
                (default 8, exact for polynomials up to degree 15).
                Cost scales as ``n_gauss^d`` per consistent order of
                kinked time pairs; bump to 12-20 for stiff integrands
                at large ``t_final``.
            gl_chunk_size: positive maximum GL nodes per batch (default
                4096), or ``None`` for the eager grid at any size.  A grid of
                at most this many nodes is the eager grid; a larger one is
                evaluated in blocks, which bounds temporary batch memory and
                keeps every quadrature node and causal/kink piece. Used only
                by ``method='gauss_legendre'``.
            qmc_coupling_chunk_size: optional positive dynamic coupling batch
                bound for vectorized QMC. Raw callbacks must be sample-separable
                with stable dtype and trailing shape. ``None`` keeps the full
                batch. R/C and final statistics remain whole; see
                :doc:`/user_guide/workflow` for memory and callback limits.
        """
        from .result import Result
        from sft_wick.evaluate import _execution_options

        execution = _execution_options(
            propagator_cache_bytes,
            propagator_cache_contract,
            contraction_backend,
            contraction_max_bytes,
        )

        # Collect tasks (diagram_term + metadata) up-front, in stable order,
        # then dispatch the whole batch to ``integrate_diagrams`` which handles
        # the sequential vs joblib loky path internally (n_jobs=1 stays serial,
        # bit-identical to the pre-refactor loop).
        if _prepared is None:
            orders_list, _vtype_filter, tasks = self._selection(
                orders, vertex_types, "Expansion.evaluate"
            )
        else:
            orders_list = self._orders_list(orders, "Expansion.evaluate")
            tasks = _prepared.tasks
        coupling_values = (
            self.system.build_coupling_values()
            if _prepared is None
            else _prepared.coupling_values
        )
        component_pair = _component_tuple(
            component_pair,
            self.observable_repr,
            self.system.n_components,
            "Expansion.evaluate",
        )
        fi = _component_indices(component_pair, self.observable_repr)

        diagram_terms = [task[3] for task in tasks]

        from sft_wick.evaluate import integrate_diagrams

        # A cache built outside `System.propagators()` (see
        # `propagators_from_cache`) carries no record of which system it was
        # meant for.  An UNDER-counted one silently returns a wrong number
        # rather than failing, so check here, where both counts are known.
        _cache_model = getattr(getattr(propagators, "cache", None), "model", None)
        _cache_n = getattr(_cache_model, "n_components", None)
        if _cache_n is not None and int(_cache_n) != int(self.system.n_components):
            raise ValueError(
                f"the propagator cache has n_components={_cache_n} but the "
                f"expansion's system has {self.system.n_components}.  A "
                f"mismatched cache does not fail on its own -- it silently "
                f"returns a wrong number."
            )

        # Every time integral starts at the system's t_min, from which the
        # propagators were built.  Up to 0.4.2 t_min was not passed on and
        # the integrals started at 0.
        t_min = float(self.system.t_min)
        _cache_t_min = getattr(_cache_model, "t_min", None)
        if _cache_t_min is not None and float(_cache_t_min) != t_min:
            raise ValueError(
                f"the propagator cache was built with t_min={_cache_t_min} "
                f"but the expansion's system has t_min={t_min}."
            )

        _guard_external_times(
            self,
            propagators,
            external_times,
            "Expansion.evaluate",
        )
        _guard_single_site(self, diagram_terms, positions)

        _total, details = integrate_diagrams(
            diagram_terms,
            coupling_values=coupling_values,
            lambda_f=t_final,
            t_min=t_min,
            cache=propagators.cache if _lookup_cache is None else _lookup_cache,
            **execution,
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
            _prepared_integrands=None if _prepared is None else _prepared.integrands,
            gl_chunk_size=gl_chunk_size,
            **(
                {}
                if qmc_coupling_chunk_size is None
                else {"qmc_coupling_chunk_size": qmc_coupling_chunk_size}
            ),
        )

        per_diagram = []
        per_order: dict[int, float] = defaultdict(float)
        per_vtype: dict[str, float] = defaultdict(float)
        total = 0.0
        # Zero is a computed coefficient at an order no diagram reaches.
        per_order.update({order: 0.0 for order in self._zero_orders(orders_list)})
        for task_idx, ((order, i, vtype, dt), (val, err)) in enumerate(
            zip(tasks, details)
        ):
            per_diagram.append(
                {
                    "order": order,
                    "diagram_idx": i,
                    "vertex_type": vtype,
                    "n_cross_C": (
                        count_cross_group_c(dt)
                        if _prepared is None
                        else _prepared.cross_counts[task_idx]
                    ),
                    "value": val,
                    "error": err,
                }
            )
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
        component_pairs: Iterable[tuple] | None = None,
        component_tuples: Iterable[tuple] | None = None,
        orders: int | Iterable[int] | None = None,
        vertex_types: str | Iterable[str] | None = None,
        integrate_over: Any = None,
        method: str = "qmc_vectorized",
        n_samples: int = 2**13,
        seed: int | None = 42,
        n_jobs: int = 1,
        evaluate_n_jobs: int = 1,
        n_gauss: int = 8,
        progress: Any = None,
        reuse_integrands: bool = False,
        gl_chunk_size: int | None = DEFAULT_GL_CHUNK_SIZE,
        qmc_coupling_chunk_size: int | None = None,
        propagator_cache_bytes: int | None = None,
        propagator_cache_contract: str | None = None,
        contraction_backend: str = "ordered",
        contraction_max_bytes: int = DEFAULT_CONTRACTION_MAX_BYTES,
    ):
        """Cartesian-product sweep over positions, t_final, external
        times and component tuples.

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
            component_pairs: the older name of ``component_tuples``,
                kept for 2-point observables; give one or the other.
            component_tuples: list of component-index tuples, one index
                per observable operator in order: ``(a, b, c)`` for
                ``<phi_a(x) phi_b(y) phi_c(z)>``.  The tuples are a
                further Cartesian axis, and each fills the component
                columns of the result rows, named by position: ``a``,
                ``b``, ``c``, ... whatever index letters the observable
                uses (a 2-point sweep keeps its ``a``, ``b`` columns).
                ``None`` for both arguments sweeps the single tuple
                ``(0, ..., 0)``.  A tuple of the wrong length, an index
                outside ``0..N-1``, a repeated tuple, or a tuple giving two
                operators that share a component letter two different
                indices raises ``ValueError``.
            orders, vertex_types: optional filters with the same semantics
                as in :meth:`evaluate`; only diagrams of these orders whose
                :meth:`_vertex_type_label` lies in the label set are
                integrated.  ``None`` ⇒ all orders and channels.  Both are
                checked before any grid point is integrated.  A selected
                order at which the expansion holds no diagram gets one row
                per grid point with ``value`` 0 and null diagram columns
                (under the connected target, every selected order no
                selected diagram reaches).
            n_jobs: parallelise over Cartesian-product grid points
                (``positions × t_final × external times × component
                tuples``).
                ``1`` (default) preserves the original sequential
                behaviour, bit-identical when seed is fixed.
                ``-1`` uses all CPU cores via joblib loky.
            evaluate_n_jobs: parallelise over diagrams **inside** each
                grid point's :meth:`evaluate` call.  Mutually
                exclusive with ``n_jobs > 1`` (nested loky pools are
                not supported); the dispatcher raises if both are
                set.  Use ``n_jobs > 1`` when the sweep grid is
                large; use ``evaluate_n_jobs > 1`` when each grid
                point has many diagrams (typical at orders >= 2).
            method, n_samples, seed, n_gauss: integrator knobs;
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
            reuse_integrands: opt in to preparing static coupling tensors
                and spatial analysis once per component tuple for this sweep.
                Dynamic couplings use the ordinary preparation path.  Bindings
                are discarded on return; later sweeps observe updated tensors.
                Trades memory for reduced preparation time, including under
                either supported joblib parallelism mode.  Default ``False``.
            gl_chunk_size: positive GL batch bound (default 4096) or
                ``None``, with the same semantics as in :meth:`evaluate`.
                Each worker has its own batch arrays; the total memory
                depends on worker count.
            qmc_coupling_chunk_size: optional dynamic coupling batch bound,
                with the same contract as in :meth:`evaluate`. Each worker
                has its own arrays; this does not bound total process memory.

        Returns:
            :class:`SweepResult` with a pandas-friendly tidy table.
        """
        from sft_wick.evaluate import _execution_options

        execution = _execution_options(
            propagator_cache_bytes,
            propagator_cache_contract,
            contraction_backend,
            contraction_max_bytes,
        )
        from sft_wick.progress import progress as _progress_scope

        with _progress_scope(progress):
            return self._sweep(
                propagators,
                **execution,
                positions_grid=positions_grid,
                t_final_grid=t_final_grid,
                external_times_grid=external_times_grid,
                component_pairs=component_pairs,
                component_tuples=component_tuples,
                orders=orders,
                vertex_types=vertex_types,
                integrate_over=integrate_over,
                method=method,
                n_samples=n_samples,
                seed=seed,
                n_jobs=n_jobs,
                evaluate_n_jobs=evaluate_n_jobs,
                n_gauss=n_gauss,
                reuse_integrands=reuse_integrands,
                gl_chunk_size=gl_chunk_size,
                qmc_coupling_chunk_size=qmc_coupling_chunk_size,
            )

    def _sweep(
        self,
        propagators,
        *,
        positions_grid,
        t_final_grid,
        external_times_grid,
        component_pairs,
        component_tuples,
        orders,
        vertex_types,
        integrate_over,
        method,
        n_samples,
        seed,
        n_jobs,
        evaluate_n_jobs,
        n_gauss,
        reuse_integrands,
        gl_chunk_size,
        qmc_coupling_chunk_size=None,
        propagator_cache_bytes: int | None = None,
        propagator_cache_contract: str | None = None,
        contraction_backend: str = "ordered",
        contraction_max_bytes: int = DEFAULT_CONTRACTION_MAX_BYTES,
    ):
        from sft_wick.evaluate import _execution_options, _propagator_scope

        execution = _execution_options(
            propagator_cache_bytes,
            propagator_cache_contract,
            contraction_backend,
            contraction_max_bytes,
        )
        with _propagator_scope(
            propagators.cache, propagator_cache_bytes, propagator_cache_contract
        ) as lookup:
            return self._sweep_dispatch(
                propagators,
                _lookup_cache=None if lookup is propagators.cache else lookup,
                _execution=execution,
                positions_grid=positions_grid,
                t_final_grid=t_final_grid,
                external_times_grid=external_times_grid,
                component_pairs=component_pairs,
                component_tuples=component_tuples,
                orders=orders,
                vertex_types=vertex_types,
                integrate_over=integrate_over,
                method=method,
                n_samples=n_samples,
                seed=seed,
                n_jobs=n_jobs,
                evaluate_n_jobs=evaluate_n_jobs,
                n_gauss=n_gauss,
                reuse_integrands=reuse_integrands,
                gl_chunk_size=gl_chunk_size,
                qmc_coupling_chunk_size=qmc_coupling_chunk_size,
            )

    def _sweep_dispatch(
        self,
        propagators,
        *,
        positions_grid,
        t_final_grid,
        external_times_grid,
        component_pairs,
        component_tuples,
        orders,
        vertex_types,
        integrate_over,
        method,
        n_samples,
        seed,
        n_jobs,
        evaluate_n_jobs,
        n_gauss,
        reuse_integrands,
        gl_chunk_size,
        qmc_coupling_chunk_size=None,
        _lookup_cache=None,
        _execution=None,
    ):
        from .result import SweepResult
        from sft_wick.evaluate import (
            _validate_gl_chunk_size,
            _validate_qmc_coupling_chunk_size,
        )

        gl_chunk_size = _validate_gl_chunk_size(gl_chunk_size)
        qmc_coupling_chunk_size = _validate_qmc_coupling_chunk_size(
            qmc_coupling_chunk_size
        )

        if int(n_jobs) != 1 and int(evaluate_n_jobs) != 1:
            raise ValueError(
                "Specify exactly one of {n_jobs, evaluate_n_jobs} > 1; "
                "nested joblib loky pools are not supported."
            )
        if not isinstance(reuse_integrands, (bool, np.bool_)):
            raise TypeError("reuse_integrands must be a boolean.")

        # The diagrams each grid point integrates, checked before the grid.
        orders_list, vtype_filter, tasks = self._selection(
            orders, vertex_types, "Expansion.sweep"
        )
        zero_orders = self._zero_orders(orders_list)

        pos_keys = list(positions_grid.keys())
        pos_values = [positions_grid[k] for k in pos_keys]

        # The component axis: one index per observable operator, so a 3- or
        # 4-point observable sweeps triples or quadruples.  Up to 0.4.x the
        # grid loop unpacked every entry as a pair.
        comp_tuples = _resolve_component_tuples(
            component_pairs,
            component_tuples,
            self.observable_repr,
            self.system.n_components,
            "Expansion.sweep",
        )
        comp_cols = component_columns(len(self.observable_repr))

        # ``external_times_grid`` mirrors ``positions_grid``: one list per
        # external point, swept as a further Cartesian axis.  This is what
        # makes two-time observables (R(t, t') and C(t, t'), the DMFT order
        # parameters) reachable declaratively; with every external pinned at
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
        # A value listed twice on one axis puts the same grid point in the
        # sweep twice, and `totals()` then adds the two copies together; an
        # empty axis gives no rows.
        _check_axis("t_final_grid", t_final_grid)
        for k, v in zip(pos_keys, pos_values):
            _check_axis(f"positions_grid[{k!r}]", v)
        for k, v in zip(et_keys, et_values):
            _check_axis(f"external_times_grid[{k!r}]", v)

        # The row dict is flat, so a column that coincides with another
        # would silently OVERWRITE it: an external named `final`
        # shadowing the sweep's own `t_final`, a spatial point literally
        # named `t_x`, or a spatial label `c` in a 3-point sweep, whose
        # third component column is `c`.
        _fixed = {"t_final", *comp_cols, *_DIAGRAM_COLUMNS}
        _clash = {f"t_{k}" for k in et_keys} & (set(pos_keys) | _fixed)
        if _clash:
            raise ValueError(
                f"external_times_grid would emit column(s) {sorted(_clash)}, "
                f"which collide with existing sweep columns and would "
                f"silently overwrite them.  Rename the affected point(s)."
            )
        _clash = set(pos_keys) & _fixed
        if _clash:
            raise ValueError(
                f"positions_grid key(s) {sorted(_clash)} coincide with sweep "
                f"column(s) of the same name (t_final, the component columns "
                f"{list(comp_cols)}, or a per-diagram column) and would be "
                f"silently overwritten in the result rows.  Rename the "
                f"spatial label(s) in the observable."
            )

        # Flatten the Cartesian product to a list of grid-point tasks.
        grid_tasks: list[tuple[dict, Any, dict, tuple]] = []
        for pos_tuple in itertools.product(*pos_values):
            positions = dict(zip(pos_keys, pos_tuple))
            for t_f in t_final_grid:
                for et_tuple in (itertools.product(*et_values) if et_keys else [()]):
                    ext_times = dict(zip(et_keys, et_tuple)) or None
                    for comp in comp_tuples:
                        grid_tasks.append((positions, t_f, ext_times, comp))

        # Warn once per distinct external-times combination rather than once
        # per grid point, which would bury the message.
        for _et in {
            tuple(sorted((ext or {}).items())) for _p, _t, ext, _c in grid_tasks
        }:
            _guard_external_times(
                self,
                propagators,
                dict(_et) or None,
                "Expansion.sweep",
            )

        # A spatially structureless cache returns the same C at every
        # separation, so a positions sweep yields a column of identical
        # numbers beside a varying position column.  That is correct for a
        # disorder-averaged single-site cache such as ``spectral_cache`` and a
        # silent mistake for anyone who expected a separation dependence.
        #
        # The question is "does this SWEEP cover more than one distinct
        # configuration of external positions", which is knowable only here,
        # from the grid.  Asking it per grid point instead (does this one
        # point put its two externals at different places?) gets both halves
        # wrong: it says nothing about a sweep over a single position key, and
        # it fires on every ordinary single-separation ``evaluate(x != y)``.
        _configs = {
            tuple(
                sorted(
                    (k, tuple(np.atleast_1d(v).ravel().tolist())) for k, v in _p.items()
                )
            )
            for _p, _t, _e, _c in grid_tasks
        }
        if len(_configs) > 1 and not _cache_has_spatial_table(
            getattr(propagators, "cache", None)
        ):
            warnings.warn(
                f"Expansion.sweep: the propagator cache has no spatial "
                f"structure, so C is the same at every separation -- the "
                f"{len(_configs)} position configurations in this sweep will "
                f"all give identical values.  That is expected for a "
                f"disorder-averaged (single-site) cache such as "
                f"`spectral_cache`; it is a mistake if you expected a "
                f"separation dependence.",
                UserWarning,
                stacklevel=2,
            )

        # Diagrams per grid point, for the progress bar's total.
        n_diag = len(tasks)
        prepared = (
            self._prepare_sweep(
                tasks,
                comp_tuples,
                **(
                    {}
                    if (_execution or {}).get("contraction_backend") != "factorized"
                    else {
                        "contraction_backend": "factorized",
                        "contraction_max_bytes": (_execution or {}).get(
                            "contraction_max_bytes", DEFAULT_CONTRACTION_MAX_BYTES
                        ),
                    }
                ),
            )
            if reuse_integrands
            else {}
        )

        def _eval_grid_point(task, tick=None):
            positions, t_f, ext_times, comp = task
            res = self.evaluate(
                propagators,
                **(_execution or {}),
                **({} if _lookup_cache is None else {"_lookup_cache": _lookup_cache}),
                positions=positions,
                t_final=t_f,
                external_times=ext_times,
                component_pair=comp,
                orders=orders_list,
                vertex_types=vtype_filter,
                integrate_over=integrate_over,
                method=method,
                n_samples=n_samples,
                seed=seed,
                n_jobs=evaluate_n_jobs,
                n_gauss=n_gauss,
                _progress_tick=tick,
                _prepared=prepared.get(comp),
                gl_chunk_size=gl_chunk_size,
                **(
                    {}
                    if qmc_coupling_chunk_size is None
                    else {"qmc_coupling_chunk_size": qmc_coupling_chunk_size}
                ),
            )
            return positions, t_f, ext_times, comp, res

        from sft_wick.progress import progress_bar

        desc = f"sweep ({len(grid_tasks)} grid points x {n_diag} diagrams)"
        with progress_bar(len(grid_tasks) * n_diag, desc, unit="diagram") as tick:
            if int(n_jobs) == 1 or len(grid_tasks) <= 2:
                # Sequential: bit-identical to the pre-refactor nested loops.
                results = [_eval_grid_point(t, tick) for t in grid_tasks]
            else:
                from joblib import Parallel, delayed

                gen = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator")(
                    delayed(_eval_grid_point)(t) for t in grid_tasks
                )
                results = []
                for r in gen:
                    results.append(r)
                    tick(n_diag)

        rows = []
        for positions, t_f, ext_times, comp, res in results:
            # Hashable normalisation: d-dim vector positions arrive as
            # ``list`` or ``np.ndarray``; pandas ``groupby`` (used in
            # :meth:`SweepResult.totals`) factorises group keys via a
            # hash table, which rejects list-typed cells with
            # ``TypeError: unhashable type: 'list'``.  Coerce here so
            # downstream aggregation works for both scalar and d-dim
            # positions.
            hashable_positions = {
                k: _hashable_position(v) for k, v in positions.items()
            }
            # One column per observable operator, named by position.
            comp_fields = dict(zip(comp_cols, comp))
            point_rows = list(res.per_diagram)
            present_orders = {row["order"] for row in point_rows}
            # Grid/order placeholders record exact zero coefficients; null
            # diagram metadata means they are not physical diagrams.
            point_rows.extend(
                {
                    "order": order,
                    "diagram_idx": None,
                    "vertex_type": None,
                    "n_cross_C": None,
                    "value": 0.0,
                    "error": 0.0,
                }
                for order in zero_orders
                if order not in present_orders
            )
            for pd_row in point_rows:
                rows.append(
                    {
                        **hashable_positions,
                        "t_final": t_f,
                        # ``t_<point>`` rather than the bare name, which is
                        # already taken by that point's spatial position.
                        **{f"t_{k}": (ext_times or {}).get(k) for k in et_keys},
                        **comp_fields,
                        **pd_row,
                    }
                )
        return SweepResult(
            rows=rows,
            position_keys=tuple(pos_keys),
            external_time_keys=tuple(f"t_{k}" for k in et_keys),
            component_keys=comp_cols,
        )

    # --------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------- #

    def _orders_list(self, orders, label):
        """The selected orders as a sorted list; every one must be an order
        of this expansion (it used to end in ``KeyError``)."""
        selected = _order_selection(orders, f"{label}: orders")
        if selected is None:
            return list(self.orders)
        absent = [o for o in selected if o not in self.dts_by_order]
        if absent:
            raise ValueError(
                f"{label}: order(s) {absent} are not in the expansion; it "
                f"holds orders {list(self.orders)}."
            )
        return list(selected)

    def _zero_orders(self, orders_list):
        """The selected orders reported as a computed zero when no selected
        diagram reaches them: every one under the connected target, and
        under the moment target those at which the expansion holds no
        diagram (an odd order of a theory with only cubic vertices).

        ``evaluate`` lists them in ``by_order`` and ``sweep`` adds a zero
        row for each, so a sweep that selects only such orders has rows.
        A moment-target sweep of them used to return no row, and
        :meth:`SweepResult.totals` then raised ``KeyError`` after the
        propagators and the sweep had run.
        """
        if self.target == "connected":
            return list(orders_list)
        return [order for order in orders_list if not self.dts_by_order[order]]

    def _selection(self, orders, vertex_types, label):
        """``(orders_list, vtype_filter, tasks)`` for an ``orders`` /
        ``vertex_types`` selection, checked before anything is integrated."""
        orders_list = self._orders_list(orders, label)
        vtype_filter = _vertex_type_selection(vertex_types, f"{label}: vertex_types")
        tasks = self._evaluation_tasks(orders_list, vtype_filter, label)
        return orders_list, vtype_filter, tasks

    def _evaluation_tasks(self, orders, vtype_filter, label="Expansion.evaluate"):
        """``(order, i, label, diagram)`` for each selected diagram.

        Under the moment target, a requested vertex type that no diagram at
        these orders carries raises ``ValueError``.  It used to select
        nothing, so ``evaluate`` returned 0 for it and a sweep that selected
        nothing else failed in ``totals()``; next to other labels a
        misspelt one dropped its channel without a message.

        The connected target reports an order with no selected diagram as
        a computed zero (``evaluate`` fills it in, ``sweep`` adds a zero
        row), so there a label is accepted when a diagram at one of these
        orders could carry it: the sorted names of at most that many
        distinct vertices of the system.  A misspelt label, a vertex the
        system lacks, or more names than the order allows still raise.
        """
        tasks = []
        present = set()
        for order in orders:
            for i, dt in enumerate(self.dts_by_order[order]):
                vtype = self._vertex_type_label(dt)
                present.add(vtype)
                if vtype_filter is None or vtype in vtype_filter:
                    tasks.append((order, i, vtype, dt))
        absent = sorted(set(vtype_filter or ()) - present)
        possible = None
        if absent and self.target == "connected":
            possible = self._possible_labels(orders)
            absent = [lbl for lbl in absent if lbl not in possible]
        if absent:
            hint = (
                ""
                if possible is None
                else f"  Under the connected target a label a diagram at "
                f"these orders could carry gives a zero coefficient; those "
                f"labels are {sorted(possible)}."
            )
            raise ValueError(
                f"{label}: no diagram at order(s) {list(orders)} has the "
                f"vertex type(s) {absent}; the vertex types present are "
                f"{sorted(present)}.  A label joins the sorted names of a "
                f"diagram's distinct vertices: two F vertices give 'F', an F "
                f"and a K give 'FK', and an order-0 diagram gives ''.{hint}"
            )
        return tuple(tasks)

    def _possible_labels(self, orders) -> set:
        """Every label a diagram at one of ``orders`` could carry: ``''`` at
        order 0, else the sorted, joined names of at most ``max(orders)``
        distinct vertices (each vertex is one order)."""
        names = sorted(self.system.build_coupling_values())
        out = {""} if 0 in orders else set()
        for k in range(1, min(max(orders, default=0), len(names)) + 1):
            out.update("".join(c) for c in itertools.combinations(names, k))
        return out

    def _prepare_sweep(
        self,
        tasks,
        component_tuples,
        *,
        contraction_backend="ordered",
        contraction_max_bytes=DEFAULT_CONTRACTION_MAX_BYTES,
    ):
        """Bind static diagrams only; no callable results or global cache.

        System specs contain mutable arrays despite their frozen wrappers.
        Reading MSR couplings here on every sweep avoids stale cross-call hits.
        Exact DiagramTerms retain ordered slots, aliases and absorbed R legs.
        """
        from sft_wick.evaluate import _build_execution_integrand

        cv = self.system.build_coupling_values()
        dynamic_names = {name for name, value in cv.items() if callable(value)}
        is_static = [
            not (_collect_symbol_names(dt.coupling_sum) & dynamic_names)
            for _order, _i, _vtype, dt in tasks
        ]
        cross_counts = tuple(count_cross_group_c(dt) for _o, _i, _v, dt in tasks)
        prepared = {}
        for comp in component_tuples:
            fi = _component_indices(comp, self.observable_repr)
            integrands = tuple(
                (
                    _build_execution_integrand(
                        dt, cv, fi, contraction_backend, contraction_max_bytes
                    )
                    if static
                    else None
                )
                for (_o, _i, _v, dt), static in zip(tasks, is_static)
            )
            prepared[comp] = _PreparedSweepPoint(tasks, cv, integrands, cross_counts)
        return prepared

    @staticmethod
    def _vertex_type_label(dt) -> str:
        """Extract the sorted unique coupling-symbol names from
        ``dt.coupling_sum`` and join them into a single string
        (matches demo2's convention: ``{'F'}`` → ``"F"``,
        ``{'F', 'K'}`` → ``"FK"``).  :class:`System` refuses vertex
        names for which two sets would get one label (``F``, ``K`` and
        ``FK``), so the label names one set of vertices."""
        names = _collect_symbol_names(dt.coupling_sum)
        if not names:
            return ""  # order-0 trivial
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
    Operators sharing a letter write one entry; :func:`_component_tuple`
    has checked that they were given one index.
    """
    fi: dict[str, int] = {}
    for i, (name, comp, spatial) in enumerate(observable_repr):
        if comp is not None and i < len(component_pair):
            fi[comp] = int(component_pair[i])
    return fi


#: Per-diagram columns of a sweep row (the keys of ``Result.per_diagram``).
_DIAGRAM_COLUMNS = (
    "order",
    "diagram_idx",
    "vertex_type",
    "n_cross_C",
    "value",
    "error",
)


def component_columns(n_operators: int) -> tuple[str, ...]:
    """Sweep-row names of the component columns of an ``n``-operator
    observable: ``a`` for the first operator, ``b`` for the second, and so
    on, whatever index letters the observable itself uses.  A 2-point sweep
    therefore keeps its ``a``, ``b`` columns."""
    letters = string.ascii_lowercase
    if n_operators > len(letters):
        raise ValueError(
            f"an observable of {n_operators} operators needs more component "
            f"columns than the {len(letters)} letters a-z."
        )
    return tuple(letters[:n_operators])


def _observable_text(observable) -> str:
    """``<phi_a(x) phi_b(y)>`` from an observable repr, or from the
    operator strings themselves."""
    parts = []
    for op in observable:
        if isinstance(op, str):
            parts.append(op)
            continue
        name, comp, spatial = op
        if comp is None:
            parts.append(f"{name}({spatial})")
        elif isinstance(comp, str):
            parts.append(f"{name}_{comp}({spatial})")
        else:
            parts.append(f"{name}_{''.join(map(str, comp))}({spatial})")
    return "<" + " ".join(parts) + ">"


def _component_tuple(component, observable, n_components, label):
    """``component`` as a tuple of ints, one per observable operator;
    ``(0, ..., 0)`` for ``None``.

    ``observable`` is the expansion's ``observable_repr`` or the operator
    strings; only its length and its text are used.
    """
    n_ops = len(observable)
    zeros = (0,) * n_ops
    if component is None:
        return zeros
    if isinstance(component, (str, bytes)) or not hasattr(component, "__len__"):
        raise ValueError(
            f"{label}: component tuple {component!r} is not a sequence of "
            f"component indices; write e.g. {zeros}."
        )
    entries = tuple(component)
    if len(entries) != n_ops:
        raise ValueError(
            f"{label}: component tuple {entries} has {len(entries)} "
            f"{'entry' if len(entries) == 1 else 'entries'}, but the "
            f"observable {_observable_text(observable)} has {n_ops} "
            f"{'operator' if n_ops == 1 else 'operators'}; give one "
            f"component index per operator, e.g. {zeros}."
        )
    n = int(n_components)
    for c in entries:
        if isinstance(c, (bool, np.bool_)) or not isinstance(c, numbers.Integral):
            raise ValueError(
                f"{label}: component index {c!r} in {entries} is not an " f"integer."
            )
        if not 0 <= int(c) < n:
            raise ValueError(
                f"{label}: component index {int(c)} in {entries} is outside "
                f"0..{n - 1} (n_components = {n})."
            )
    out = tuple(int(c) for c in entries)
    _require_one_index_per_letter(out, observable, label)
    return out


def _component_letters(observable) -> tuple:
    """The component letter of each operator of ``observable`` (an
    ``observable_repr`` or the operator strings); ``None`` for an operator
    without one, or for a string that does not parse (the parser reports
    it later)."""
    from .system import _OBS_PATTERN

    letters = []
    for op in observable:
        if isinstance(op, str):
            m = _OBS_PATTERN.match(op)
            letters.append(None if m is None else m.group("comp"))
        else:
            letters.append(op[1])
    return tuple(letters)


def _require_one_index_per_letter(component, observable, label) -> None:
    """Refuse a component tuple that gives two operators sharing one
    component letter two different indices.

    The letter names the index, and the integrand reads one value per name
    (:func:`_component_indices`), so the later operator's index used to
    replace the earlier one's: ``<phi_a(x) phi_a(y)>`` asked for ``(0, 1)``
    returned the ``(1, 1)`` value in rows labelled ``a = 0, b = 1``.  A
    shared letter given one index is the same observable as distinct
    letters given that index twice, and stays legal.
    """
    first: dict = {}
    for pos, letter in enumerate(_component_letters(observable)):
        if letter is None:
            continue
        i = first.setdefault(letter, pos)
        if component[i] != component[pos]:
            raise ValueError(
                f"{label}: operators {i + 1} and {pos + 1} of the observable "
                f"{_observable_text(observable)} share the component letter "
                f"{letter!r}, so they carry one component index, but the "
                f"component tuple {tuple(component)} gives them "
                f"{component[i]} and {component[pos]}.  Give the operators "
                f"distinct letters to ask for distinct indices."
            )


def _ambiguous_vertex_type_label(names):
    """Two distinct sets of ``names`` that the vertex-type label (the sorted
    names, concatenated; :meth:`Expansion._vertex_type_label`) does not
    tell apart, as sorted tuples, or ``None`` when the label is injective
    on the subsets of ``names``.

    Two such sets read one string as two strictly increasing sequences of
    names.  Dropping the names the two readings share at the start leaves
    two readings of a shorter string that differ in their first name, so
    one first name is a proper prefix of the other.  The search follows
    pairs of partial readings from there; a state is the name each reading
    used last and the text the shorter reading still has to cover, which
    is a suffix of a name, so there are finitely many.
    """
    names = sorted(set(names))
    queue = deque(
        ((u,), (v,), v[len(u):])
        for u in names
        for v in names
        if u != v and v.startswith(u)
    )
    seen = set()
    while queue:
        behind, ahead, rest = queue.popleft()
        state = (behind[-1], ahead[-1], rest)
        if state in seen:
            continue
        seen.add(state)
        for w in names:
            if w <= behind[-1]:
                continue
            if rest.startswith(w):
                if w == rest:
                    return tuple(sorted((behind + (w,), ahead)))
                queue.append((behind + (w,), ahead, rest[len(w):]))
            elif w.startswith(rest):
                queue.append((ahead, behind + (w,), w[len(rest):]))
    return None


def _resolve_component_tuples(
    component_pairs, component_tuples, observable, n_components, label
):
    """The component axis of a sweep, as a list of validated tuples.

    ``component_pairs`` is the older name of ``component_tuples``; ``None``
    for both gives the single tuple ``(0, ..., 0)``.
    """
    if component_pairs is not None and component_tuples is not None:
        raise ValueError(
            f"{label}: pass component_tuples or component_pairs, not both "
            f"(component_pairs is the older name of the same axis)."
        )
    given = component_tuples if component_tuples is not None else component_pairs
    zeros = (0,) * len(observable)
    if given is None:
        return [zeros]
    if isinstance(given, (str, bytes)) or not hasattr(given, "__iter__"):
        raise ValueError(
            f"{label}: the component axis must be a list of component "
            f"tuples, e.g. [{zeros}]; got {given!r}."
        )
    given = list(given)
    if not given:
        raise ValueError(
            f"{label}: the component axis is an empty list; the sweep would "
            f"return no rows."
        )
    out: list[tuple[int, ...]] = []
    for comp in given:
        if isinstance(comp, (str, bytes)) or not hasattr(comp, "__len__"):
            raise ValueError(
                f"{label}: the component axis must be a list of component "
                f"tuples, one index per observable operator, e.g. "
                f"[{zeros}]; got the entry {comp!r}."
            )
        t = _component_tuple(comp, observable, n_components, label)
        if t in out:
            raise ValueError(
                f"{label}: the component tuple {t} is listed twice; "
                f"totals() would add the repeated rows together."
            )
        out.append(t)
    return out


def _is_integer(value) -> bool:
    """An integer that is not a boolean (``bool`` is an ``Integral``)."""
    return isinstance(value, numbers.Integral) and not isinstance(
        value, (bool, np.bool_)
    )


def _order_selection(orders, label, required=False):
    """``orders`` as a sorted tuple of distinct ints; ``None`` stays ``None``.

    One integer is one order (``--override sweep.orders=2`` gives one).  A
    boolean, a string, bytes, a float or another scalar raises
    ``TypeError``, and so does a list entry that is not an integer:
    iterating ``"2"`` used to give the order 2, ``b"2"`` the order 50, and
    ``int(2.5)`` gave 2.  An empty selection and a negative order raise
    ``ValueError``.  ``label`` names the argument, e.g. ``"sweep.orders"``;
    ``required`` marks one with no "every order" to fall back on
    (``System.expand``, ``expand.orders``), which changes the advice in that
    message.
    """
    if orders is None:
        return None
    if _is_integer(orders):
        entries = [orders]
    else:
        entries = _order_entries(orders, label, required)
    negative = sorted({int(o) for o in entries if int(o) < 0})
    if negative:
        raise ValueError(
            f"{label}: the orders {negative} in {orders!r} are negative; an "
            f"order counts interaction vertices and starts at 0."
        )
    return tuple(sorted({int(o) for o in entries}))


def _order_entries(orders, label, required):
    """The entries of a non-integer ``orders``, each an integer; see
    :func:`_order_selection`."""
    usage = (
        f"{label} must be an integer order or a list of them, e.g. 2 or "
        f"[0, 2]; got {orders!r}."
    )
    if isinstance(
        orders, (str, bytes, collections.abc.Mapping, numbers.Number, np.bool_)
    ):
        raise TypeError(usage)
    try:
        entries = list(orders)
    except TypeError:
        raise TypeError(usage) from None
    for o in entries:
        if not _is_integer(o):
            raise TypeError(
                f"{label}: the order {o!r} in {orders!r} is not an integer."
            )
    if not entries:
        if required:
            advice = "give at least one order, e.g. 2 or [0, 2]"
        elif label.startswith("sweep."):
            advice = "omit the key or write null for every order of the expansion"
        else:
            advice = "pass None for every order of the expansion"
        raise ValueError(f"{label} is empty, so it selects no order; {advice}.")
    return entries


def _vertex_type_selection(vertex_types, label):
    """``vertex_types`` as a frozenset of labels; ``None`` stays ``None``.

    A single string is one label: ``set("FK")`` used to select ``{"F",
    "K"}``, so ``vertex_types="FK"`` returned the F channel.  Any other
    iterable of strings (a list, tuple or set, a NumPy array such as
    ``df["vertex_type"].unique()``, a pandas ``Index``, a generator) is a
    set of labels.  Bytes, a mapping, a scalar and an entry that is not a
    string raise ``TypeError``, and an empty selection raises
    ``ValueError``.  ``label`` names the argument, e.g.
    ``"sweep.vertex_types"``.
    """
    if vertex_types is None:
        return None
    if isinstance(vertex_types, str):
        return frozenset((str(vertex_types),))
    usage = (
        f"{label} must be a vertex-type label such as 'FK' or a list of "
        f"labels such as ['F', 'FK']; got {vertex_types!r}."
    )
    if isinstance(vertex_types, (bytes, bytearray, collections.abc.Mapping)):
        raise TypeError(usage)
    try:
        entries = list(vertex_types)
    except TypeError:  # a scalar, or a 0-d array
        raise TypeError(usage) from None
    for v in entries:
        if not isinstance(v, str):
            raise TypeError(
                f"{label}: the entry {v!r} in {vertex_types!r} is not a "
                f"vertex-type label (a string such as 'FK')."
            )
    if not entries:
        raise ValueError(
            f"{label} is empty, so it selects no diagram; pass None for "
            f"every diagram."
        )
    return frozenset(str(v) for v in entries)


def _axis_key(value):
    """A hashable stand-in for one value of a sweep axis (a scalar or
    vector position, or a time)."""
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return repr(value)
    return tuple(arr.ravel().tolist()) if arr.ndim else float(arr)


def _check_axis(name, values) -> None:
    """Refuse an empty sweep axis (no rows) and a repeated value (the same
    grid point twice, which ``totals()`` adds together)."""
    values = list(values)
    if not values:
        raise ValueError(
            f"Expansion.sweep: {name} is an empty list; the Cartesian "
            f"product would be empty and the sweep would return no rows."
        )
    seen = set()
    for v in values:
        key = _axis_key(v)
        if key in seen:
            raise ValueError(
                f"Expansion.sweep: {name} lists the value {v!r} twice; the "
                f"sweep would evaluate that grid point twice and totals() "
                f"would add the two copies together."
            )
        seen.add(key)


def _hashable_position(value):
    """A position as a hashable row cell: a vector (a list, tuple or array
    of ndim >= 1) becomes a tuple, a scalar is kept.  A numpy scalar has a
    ``tolist`` method too, so testing for that method alone turned
    ``np.float64(0.5)`` into ``tuple(0.5)``, which raises."""
    if isinstance(value, (list, tuple)):
        return tuple(value)
    if np.ndim(value) > 0:
        return tuple(np.asarray(value).tolist())
    return value
