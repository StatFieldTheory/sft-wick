"""Cost estimate for a workflow config (``sft-wick run --dry-run``).

The estimate is built from what the config implies.  The expansion is
run (the only way to know the diagram counts), the grid is enumerated,
and the C-propagator path is resolved by
:meth:`~sft_wick.workflow.Propagators.build` with the arguments the run
passes (:mod:`.config`), on a table of few nodes.  Two micro-benchmarks of
about a second in total follow: one C-propagator evaluation under the
chosen quadrature (skipped when a closed form is in use) and one diagram
evaluation per order at a single grid point, with the execution options of
the run.  The result is a wall-clock figure good to a factor of ~2,
reported BEFORE a run.  The exception is the search for a C rule at each
position when the time shape of the kernel depends on position: it is left
out of the total, measured at 160 and 600 times the printed figure, and it
runs in the dry run too when the table is a full grid.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import numpy as np

from sft_wick.evaluate import _lazy_key_separation

from .expansion import sweep_runs_serially

#: Time nodes of the table the benchmarks read.
_BENCH_TIME_NODES = 6


@dataclass(frozen=True)
class CostEstimate:
    """What a run of a config will do and roughly how long it takes."""

    diagrams_per_order: dict[int, int]
    n_grid_points: int
    n_distinct_separations: int
    sweep_method: str
    n_samples: int
    c_source: str
    is_lazy: bool
    #: Number of quadrature evaluations of C the chosen path makes.
    n_c_calls: int
    #: Measured seconds per C quadrature call (``None`` for a closed form).
    t_c_call: float | None
    #: Measured seconds per diagram evaluation at one grid point, per order.
    t_diagram_per_order: dict[int, float]
    sweep_n_jobs: int = 1
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def est_table_s(self) -> float:
        if self.t_c_call is None:
            return 0.0
        return self.n_c_calls * self.t_c_call

    @property
    def est_sweep_serial_s(self) -> float:
        per_point = sum(
            self.diagrams_per_order[o] * self.t_diagram_per_order.get(o, 0.0)
            for o in self.diagrams_per_order
        )
        return self.n_grid_points * per_point

    @property
    def est_total_serial_s(self) -> float:
        return self.est_table_s + self.est_sweep_serial_s

    def summary(self) -> str:
        n_diag = sum(self.diagrams_per_order.values())
        lines = [
            "[sft-wick] cost estimate",
            "[sft-wick]   diagrams: "
            + ", ".join(f"order {o}: {n}" for o, n in self.diagrams_per_order.items())
            + f"  (total {n_diag})",
            f"[sft-wick]   grid points: {self.n_grid_points} "
            f"(positions x t_final x external times x component tuples); "
            f"distinct separations: {self.n_distinct_separations}",
            f"[sft-wick]   integrator: {self.sweep_method}, "
            + (
                f"n_samples={self.n_samples} per diagram per grid point"
                if self.sweep_method.startswith("qmc")
                else "deterministic nodes"
            ),
            f"[sft-wick]   C propagator: {self.c_source}"
            + (" (lazy per-separation table)" if self.is_lazy else ""),
        ]
        if self.t_c_call is None:
            lines.append("[sft-wick]   C quadrature calls: 0 (closed form)")
        else:
            lines.append(
                f"[sft-wick]   C quadrature calls: {self.n_c_calls} x "
                f"{1e3 * self.t_c_call:.1f} ms = {self.est_table_s:.0f} s"
            )
        per = ", ".join(
            f"order {o}: {1e3 * t:.1f} ms" for o, t in self.t_diagram_per_order.items()
        )
        lines.append(f"[sft-wick]   per-diagram evaluation (measured): {per}")
        lines.append(
            f"[sft-wick]   sweep: {self.n_grid_points} x {n_diag} diagram "
            f"evaluations = {self.est_sweep_serial_s:.0f} s serial"
            + (
                ""
                if sweep_runs_serially(self.sweep_n_jobs, self.n_grid_points)
                else f" (~{self.est_sweep_serial_s / self._n_workers():.0f} s "
                f"on {self._n_workers()} workers)"
            )
        )
        lines.append(
            f"[sft-wick]   ROUGH TOTAL: {_fmt_duration(self.est_total_serial_s)} "
            f"serial (factor ~2 uncertainty)"
        )
        for n in self.notes:
            lines.append(f"[sft-wick]   note: {n}")
        return "\n".join(lines)

    def _n_workers(self) -> int:
        from joblib import effective_n_jobs

        return effective_n_jobs(int(self.sweep_n_jobs))


def _fmt_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def _distinct_separations(positions_grid: dict) -> int:
    keys = list(positions_grid)
    seps: set = {0.0}
    for combo in itertools.product(*(positions_grid[k] for k in keys)):
        pts = [np.atleast_1d(np.asarray(c, dtype=float)) for c in combo]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                seps.add(_lazy_key_separation(pts[i], pts[j]))
    return len(seps)


def estimate_cost(cfg, bench_seconds: float = 1.0) -> CostEstimate:
    """Estimate the cost of ``run_workflow(cfg)`` without running it."""
    from sft_wick.progress import progress as _progress_scope

    with _progress_scope(False):
        return _estimate(cfg, bench_seconds)


def _few_nodes(kwargs: dict) -> dict:
    """The run's :meth:`System.propagators` arguments on a table of few
    nodes, in this process and with no cache file: the range, homogeneity,
    interpolation and C path stay, and the build takes no time."""
    from ._run_checks import _TABLE_NODES

    nodes = max(2, _TABLE_NODES.get(kwargs["interp_method"], 2))
    few = dict(kwargs, n_jobs=1, cache_path=None)
    few["n_grid_t"] = min(kwargs["n_grid_t"], max(nodes, _BENCH_TIME_NODES))
    for key in ("n_grid_r", "n_grid_cos", "n_grid_x"):
        if kwargs[key] is not None:
            few[key] = min(kwargs[key], nodes)
    return few


def _estimate(cfg, bench_seconds) -> CostEstimate:
    from ._run_checks import check_after_expansion
    from .config import (
        _load_c_closed_form,
        build_system,
        evaluate_kwargs,
        expand_kwargs,
        propagators_kwargs,
    )
    from .resources import grid_points

    system = build_system(cfg.system)
    expansion = system.expand(**expand_kwargs(cfg))
    sw = cfg.sweep
    # The selection the sweep will make, refused as the run would refuse it.
    orders, vt, tasks = check_after_expansion(cfg, expansion)
    diagrams_per_order = {o: sum(1 for task in tasks if task[0] == o) for o in orders}
    # The benchmark evaluates one order at a time, so it passes only the
    # labels that order carries; a label of another order would be refused.
    labels_at = {
        o: sorted({task[2] for task in tasks if task[0] == o}) for o in orders
    }
    # The component axis, checked against the observable exactly as the
    # sweep will check it, so a dry run reports a wrong tuple length.
    from .expansion import _resolve_component_tuples

    comps = _resolve_component_tuples(
        None,
        sw.component_tuples,
        expansion.observable_repr,
        system.n_components,
        "sweep.component_tuples",
    )
    n_grid = grid_points(sw)
    n_sep = _distinct_separations(sw.positions_grid)

    # Resolve the C path as the run would.
    pc = cfg.propagators
    few_nodes = _few_nodes(propagators_kwargs(cfg, _load_c_closed_form(pc)))
    props = system.propagators(**few_nodes)
    notes: list[str] = []
    cache = props.cache
    if props.c_source.startswith("closed_form") or pc.c_closed_form_only:
        n_c_calls, t_c = 0, None
    else:
        n_cells = pc.n_grid_t**2
        # Half the cells when the table is fixed by its t2 >= t1 half.
        if not cache._need_earlier_first(props.homogeneity):
            n_cells = pc.n_grid_t * (pc.n_grid_t + 1) // 2
        if props.is_lazy:
            n_tables = 1 if cache._lazy_spatial_factor() is not None else n_sep
            if n_tables > 1:
                notes.append(
                    f"lazy table rebuilt for each of the {n_sep} "
                    f"distinct separations (kernel not separable); the C "
                    f"rule shown is the origin's, and the rule of each "
                    f"separation is worked out in the run, which is not in "
                    f"this estimate and can take minutes a separation"
                )
        else:
            n_tables = int(pc.n_grid_r or pc.n_grid_cos or (pc.n_grid_x or 1) ** 2)
            if (
                n_tables > 1
                and pc.c_method == "auto"
                and cache._c_time_shape_depends_on_position()
            ):
                notes.append(
                    f"the C rule is checked at each of the {n_tables} "
                    f"positions of the grid, which is not in this estimate "
                    f"and can take minutes a position; this dry run has "
                    f"done that for a grid of few nodes"
                )
        n_c_calls = n_cells * n_tables
        n1, n2 = cache._probe_positions()
        if cache.c_method_resolved == "auto":
            # The origin settled nothing: time a position the sweep reads.
            first = [np.asarray(v[0], dtype=float) for v in sw.positions_grid.values()]
            n1, n2 = first[0], first[-1]
            kw = cache._direct_kwargs(*cache._c_rule(n1, n2, pc.t_max))
        else:
            kw = cache._direct_kwargs(*cache.resolve_c_method(pc.t_max))
        mid = 0.5 * (system.t_min + pc.t_max)
        # Warm-up, then time the deep cell, slowest for adaptive rules.
        cache._C_value_direct(n1, mid, n2, mid, **kw)
        t0 = time.perf_counter()
        reps = 0
        while time.perf_counter() - t0 < 0.4 * bench_seconds:
            cache._C_value_direct(n1, pc.t_max, n2, pc.t_max, **kw)
            reps += 1
        t_c = (time.perf_counter() - t0) / reps
        if pc.n_jobs != 1:
            notes.append(
                f"propagators.n_jobs={pc.n_jobs}: table build "
                f"parallelises across cells"
            )

    # One diagram per order at one grid point, on a small throw-away
    # table so a slow quadrature does not dominate the benchmark.  An
    # explicit rule: under 'auto' the table would run the selection again.
    bench_method = {
        "gauss_legendre": "gauss_legendre",
        "graded": "gauss_legendre",
        "dblquad": "dblquad",
        "auto": "dblquad",
    }.get(cache.c_method_resolved, pc.c_method)
    bench_props = system.propagators(
        **dict(
            few_nodes, c_method=bench_method, c_n_gauss=min(pc.c_n_gauss, 12)
        )
    )
    n_bench = min(int(sw.n_samples), 2048)
    pos = {k: v[0] for k, v in sw.positions_grid.items()}
    t_f = float(max(sw.t_final_grid))
    et = (
        {k: v[0] for k, v in sw.external_times_grid.items()}
        if sw.external_times_grid
        else None
    )
    t_diag: dict[int, float] = {}

    def _evaluate(o, n):
        return expansion.evaluate(
            bench_props,
            positions=pos,
            t_final=t_f,
            external_times=et,
            component_pair=comps[0],
            orders=[o],
            vertex_types=None if vt is None else labels_at[o],
            n_samples=n,
            **evaluate_kwargs(cfg),
        )

    # Warm up on the cheapest order: the first evaluation at a separation
    # also builds that separation's lazy C table, which is a one-off cost
    # that must not be charged to every diagram.
    warm = next((o for o in orders if diagrams_per_order[o] > 0), None)
    if warm is not None:
        _evaluate(warm, 64)
    for o in orders:
        if diagrams_per_order[o] == 0:
            t_diag[o] = 0.0
            continue
        t0 = time.perf_counter()
        res = _evaluate(o, n_bench)
        dt = (time.perf_counter() - t0) / max(1, len(res.per_diagram))
        if sw.method.startswith("qmc"):
            dt *= int(sw.n_samples) / n_bench
        t_diag[o] = dt
    return CostEstimate(
        diagrams_per_order=diagrams_per_order,
        n_grid_points=n_grid,
        n_distinct_separations=n_sep,
        sweep_method=sw.method,
        n_samples=int(sw.n_samples),
        c_source=props.c_source,
        is_lazy=props.is_lazy,
        n_c_calls=n_c_calls,
        t_c_call=t_c,
        t_diagram_per_order=t_diag,
        sweep_n_jobs=int(sw.n_jobs),
        notes=tuple(notes),
    )
