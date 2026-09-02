"""Cost estimate for a workflow config (``sft-wick run --dry-run``).

The estimate is built from what the config actually implies -- the
expansion is run (it is the only way to know the diagram counts), the
grid is enumerated, the C-propagator path is resolved exactly as
:meth:`~sft_wick.workflow.Propagators.build` would resolve it -- and two
micro-benchmarks of about a second in total: one C-propagator evaluation
under the chosen quadrature (skipped when a closed form is in use) and
one diagram evaluation per order at a single grid point.  The result is
a rough wall-clock figure, good to a factor of ~2, that tells a user
BEFORE a run whether it is a coffee or an overnight job -- which is the
question the package used to answer only by staying silent for an hour.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


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
            f"(positions x t_final x external times x component pairs); "
            f"distinct separations: {self.n_distinct_separations}",
            f"[sft-wick]   integrator: {self.sweep_method}, "
            + (f"n_samples={self.n_samples} per diagram per grid point"
               if self.sweep_method.startswith("qmc") else "deterministic nodes"),
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
            + (f" (~{self.est_sweep_serial_s / max(1, self._n_workers()):.0f} s "
               f"on {self._n_workers()} workers)" if self.sweep_n_jobs != 1 else "")
        )
        lines.append(
            f"[sft-wick]   ROUGH TOTAL: {_fmt_duration(self.est_total_serial_s)} "
            f"serial (factor ~2 uncertainty)"
        )
        for n in self.notes:
            lines.append(f"[sft-wick]   note: {n}")
        return "\n".join(lines)

    def _n_workers(self) -> int:
        import os
        if self.sweep_n_jobs == -1:
            return os.cpu_count() or 1
        return max(1, int(self.sweep_n_jobs))


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
                seps.add(round(float(np.linalg.norm(pts[i] - pts[j])), 10))
    return len(seps)


def estimate_cost(cfg, bench_seconds: float = 1.0) -> CostEstimate:
    """Estimate the cost of ``run_workflow(cfg)`` without running it."""
    from sft_wick.progress import progress as _progress_scope

    from .config import _load_c_closed_form, build_system

    with _progress_scope(False):
        return _estimate(cfg, bench_seconds, build_system, _load_c_closed_form)


def _estimate(cfg, bench_seconds, build_system, load_c) -> CostEstimate:
    system = build_system(cfg.system)
    expand_diag_C = cfg.expand.diag_C and cfg.propagators.diag_C
    expansion = system.expand(
        observable=cfg.expand.observable, orders=cfg.expand.orders,
        response_phase=cfg.expand.response_phase, ito=cfg.expand.ito,
        collect_topology=cfg.expand.collect_topology, iso_R=cfg.expand.iso_R,
        diag_R=cfg.expand.diag_R, diag_C=expand_diag_C, iso_C=cfg.expand.iso_C,
        cache_path=cfg.expand.cache_path,
    )
    sw = cfg.sweep
    orders = (sorted(set(int(o) for o in sw.orders))
              if sw.orders is not None else list(expansion.orders))
    vt = None if sw.vertex_types is None else set(sw.vertex_types)
    diagrams_per_order = {
        o: sum(1 for dt in expansion.diagrams(o)
               if vt is None or expansion._vertex_type_label(dt) in vt)
        for o in orders
    }
    n_pos = int(np.prod([len(v) for v in sw.positions_grid.values()]))
    n_et = int(np.prod([len(v) for v in (sw.external_times_grid or {}).values()])) \
        if sw.external_times_grid else 1
    n_grid = n_pos * len(sw.t_final_grid) * n_et * len(sw.component_pairs)
    n_sep = _distinct_separations(sw.positions_grid)

    # Resolve the C path exactly as the run would (lazy mode builds
    # nothing until the first lookup, so this is cheap).
    pc = cfg.propagators
    c_fn = load_c(pc)
    props = system.propagators(
        t_max=pc.t_max, n_grid_t=pc.n_grid_t, homogeneity=pc.homogeneity,
        r_max=pc.r_max, n_grid_r=pc.n_grid_r, n_grid_cos=pc.n_grid_cos,
        x_max=pc.x_max, n_grid_x=pc.n_grid_x, n_jobs=1, c_closed_form=c_fn,
        interp_method=pc.interp_method, c_closed_form_only=pc.c_closed_form_only,
        c_closed_form_vectorized=pc.c_closed_form_vectorized,
        c_method=pc.c_method, c_n_gauss=pc.c_n_gauss, diag_C=pc.diag_C,
    )
    notes: list[str] = []
    cache = props.cache
    if props.c_source.startswith("closed_form") or pc.c_closed_form_only:
        n_c_calls, t_c = 0, None
    else:
        n_cells = pc.n_grid_t ** 2
        if cache._c_time_symmetric():
            n_cells = pc.n_grid_t * (pc.n_grid_t + 1) // 2
        if props.is_lazy:
            n_tables = 1 if cache._lazy_spatial_factor() is not None else n_sep
            if n_tables > 1:
                notes.append(f"lazy table rebuilt for each of the {n_sep} "
                             f"distinct separations (kernel not separable)")
        else:
            n_tables = int(pc.n_grid_r or pc.n_grid_cos or (pc.n_grid_x or 1) ** 2)
        n_c_calls = n_cells * n_tables
        method, n_g = cache.resolve_c_method(pc.t_max)
        kw = cache._direct_kwargs(method, n_g)
        mid = 0.5 * (system.t_min + pc.t_max)
        n1, n2 = cache._probe_positions()
        # Warm-up, then time the deep cell -- the slowest for adaptive rules.
        cache._C_value_direct(n1, mid, n2, mid, **kw)
        t0 = time.perf_counter()
        reps = 0
        while time.perf_counter() - t0 < 0.4 * bench_seconds:
            cache._C_value_direct(n1, pc.t_max, n2, pc.t_max, **kw)
            reps += 1
        t_c = (time.perf_counter() - t0) / reps
        if pc.n_jobs != 1:
            notes.append(f"propagators.n_jobs={pc.n_jobs}: table build "
                         f"parallelises across cells")

    # One diagram per order at one grid point, on a small throw-away
    # table so a slow quadrature does not dominate the benchmark.
    bench_props = system.propagators(
        t_max=pc.t_max, n_grid_t=6, n_jobs=1, c_closed_form=c_fn,
        c_closed_form_only=pc.c_closed_form_only,
        c_closed_form_vectorized=pc.c_closed_form_vectorized,
        c_method=pc.c_method, c_n_gauss=min(pc.c_n_gauss, 12), diag_C=pc.diag_C,
    )
    n_bench = min(int(sw.n_samples), 2048)
    pos = {k: v[0] for k, v in sw.positions_grid.items()}
    t_f = float(max(sw.t_final_grid))
    et = ({k: v[0] for k, v in sw.external_times_grid.items()}
          if sw.external_times_grid else None)
    t_diag: dict[int, float] = {}

    def _evaluate(o, n):
        return expansion.evaluate(
            bench_props, positions=pos, t_final=t_f, external_times=et,
            component_pair=tuple(sw.component_pairs[0]), orders=[o],
            vertex_types=sw.vertex_types, integrate_over=sw.integrate_over,
            method=sw.method, n_samples=n, seed=sw.seed, n_gauss=sw.n_gauss,
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
        diagrams_per_order=diagrams_per_order, n_grid_points=n_grid,
        n_distinct_separations=n_sep, sweep_method=sw.method,
        n_samples=int(sw.n_samples), c_source=props.c_source,
        is_lazy=props.is_lazy, n_c_calls=n_c_calls, t_c_call=t_c,
        t_diagram_per_order=t_diag, sweep_n_jobs=int(sw.n_jobs),
        notes=tuple(notes),
    )
