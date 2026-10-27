"""Configuration-only feedback for workflow resource controls.

This report describes eligibility, not observed execution or memory use. It
does not build diagrams, integrands or caches, or invoke user callbacks.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import WorkflowConfig


def resource_controls(cfg: WorkflowConfig) -> list[dict]:
    """Return requested values, scopes, statuses and reasons for resource controls.

    ``conditional`` means runtime structure is needed to determine use.
    ``ignored`` identifies a configuration-known bypass. No status promises
    an observed speedup or a process memory ceiling.
    """
    sw = cfg.sweep
    vertices = cfg.system.vertices + cfg.system.nonlocal_vertices
    dynamic = any(callable(v["coupling"]) for v in vertices)
    axes = list(sw.positions_grid.values()) + [sw.t_final_grid, sw.component_tuples]
    axes += list((sw.external_times_grid or {}).values())
    grid_points = math.prod(len(axis) for axis in axes)
    out = []

    def add(name, requested, scope, status, reason):
        out.append(
            dict(
                name=name,
                requested=requested,
                scope=scope,
                status=status,
                reason=reason,
            )
        )

    gl = sw.gl_chunk_size
    status = "disabled" if gl is None else "conditional"
    reason = (
        "whole-grid path requested"
        if gl is None
        else (
            "blocks GL coordinates and evaluated factors; diagrams with no integration "
            "variables bypass it; an integration piece of at most this many nodes "
            "is evaluated as the eager grid"
        )
    )
    if gl is not None and sw.method != "gauss_legendre":
        status, reason = "ignored", "method is not gauss_legendre"
    add("sweep.gl_chunk_size", gl, "GL integration workspace", status, reason)

    qmc = sw.qmc_coupling_chunk_size
    status = "disabled" if qmc is None else "conditional"
    reason = (
        "whole coupling batch requested"
        if qmc is None
        else (
            "requires vectorized QMC and a participating sample-separable dynamic "
            "coupling with stable dtype/shape; zero-dimensional diagrams bypass it; "
            "full coordinates, R/C and contracted output remain allocated"
        )
    )
    if qmc is not None:
        if sw.method not in {"qmc", "qmc_vectorized"}:
            status, reason = (
                "ignored",
                "method does not use vectorized QMC coupling batches",
            )
        elif qmc >= sw.n_samples:
            status, reason = (
                "ignored",
                "cap is at least n_samples; whole coupling batch retained",
            )
        elif not dynamic:
            status, reason = "ignored", "all configured couplings are static"
    add(
        "sweep.qmc_coupling_chunk_size",
        qmc,
        "dynamic coupling construction only",
        status,
        reason,
    )

    add(
        "sweep.reuse_integrands",
        sw.reuse_integrands,
        "static preparation per component tuple within one sweep",
        "conditional" if sw.reuse_integrands else "disabled",
        (
            "retains prepared static tensors; dynamic diagrams use ordinary preparation; "
            "a short sweep may not amortize preparation"
            if sw.reuse_integrands
            else "prepare each grid point independently"
        ),
    )

    cache_bytes = sw.propagator_cache_bytes
    status = "conditional" if cache_bytes else "disabled"
    reason = (
        "fixed-model promise required; exact native batch lookups only; retained "
        "raw key/output bytes capped, Python metadata, transient copies and tables excluded; "
        "serial sweep/evaluate lifetime, empty serialized bank and worker task/batch-local copies"
        if cache_bytes
        else "propagator batch lookup reuse disabled"
    )
    if cache_bytes and sw.propagator_cache_contract != "fixed_model":
        status, reason = (
            "invalid",
            "positive budget requires the fixed_model caller promise",
        )
    elif cache_bytes and sw.method in {"qmc_scalar", "nquad"}:
        status, reason = (
            "ignored",
            "method does not use cached propagator batch methods",
        )
    add(
        "sweep.propagator_cache_bytes",
        cache_bytes,
        "retained propagator lookup payload per owning query/worker",
        status,
        reason,
    )
    add(
        "sweep.propagator_cache_contract",
        sw.propagator_cache_contract,
        "caller promise throughout each complete query",
        "declared" if sw.propagator_cache_contract == "fixed_model" else "disabled",
        "not a callback purity test; model, tables, parameters and lookup results remain fixed",
    )

    backend = sw.contraction_backend
    status = "conditional" if backend in {"einsum", "factorized"} else "disabled"
    reason = (
        "native static dense float64 factors and exact coefficient phase only; "
        "requires complete-result tolerance validation; unsupported/budget/range cases fall back; "
        "existing coefficient preparation remains"
        if backend in {"einsum", "factorized"}
        else "ordered component assignment fold"
    )
    if backend == "factorized":
        reason = (
            "snapshotted native static monomial factors; avoid dense coefficient preparation "
            "only when admitted; explicit rounding validation and array-payload budget; "
            "oversized dense fallback fails before allocation"
        )
    if backend == "einsum" and sw.method in {"qmc_scalar", "nquad"}:
        status, reason = "ignored", "method does not use the static batch contraction"
    if backend == "factorized" and sw.method in {"qmc_scalar", "nquad"}:
        reason = (
            "method materializes the static coefficient instead of batch fusion; "
            "native preparation and materialization budget remain active; "
            "oversized known dense payload raises before allocation"
        )
    add(
        "sweep.contraction_backend",
        backend,
        "static preparation and batch contraction",
        status,
        reason,
    )
    add(
        "sweep.contraction_max_bytes",
        sw.contraction_max_bytes,
        "conservative array-payload admission; factorized also bounds dense restoration",
        (
            "conditional"
            if backend in {"einsum", "factorized"} and status != "ignored"
            else "ignored"
        ),
        (
            "not process RSS or private BLAS allocation; unknown dynamic/custom shapes are not bounded; "
            "factorized public array access or dense fallback can raise above this budget"
            if backend == "factorized"
            else "not process RSS or private BLAS allocation; ordered fallbacks remain available"
        ),
    )

    conflicting = cfg.expand.n_jobs != 1 and sw.n_jobs != 1
    for name, jobs, scope in (
        (
            "expand",
            cfg.expand.n_jobs,
            "diagrams within each sweep point (not symbolic expansion)",
        ),
        ("propagators", cfg.propagators.n_jobs, "eager C-table cells"),
        ("sweep", sw.n_jobs, f"{grid_points} Cartesian grid points"),
    ):
        status = "serial" if jobs == 1 else "conditional"
        reason = (
            "one process requested"
            if jobs == 1
            else (
                "runtime workload determines workers; each worker can retain its own "
                "model, caches and workspaces; more workers do not guarantee less time"
            )
        )
        if conflicting and name in {"expand", "sweep"}:
            status, reason = (
                "invalid",
                "expand.n_jobs and sweep.n_jobs cannot both request parallelism",
            )
        elif name == "sweep" and jobs != 1 and grid_points <= 2:
            status, reason = (
                "ignored",
                "at most two grid points: sweep executes serially",
            )
        elif name == "propagators" and jobs != 1 and cfg.propagators.c_closed_form_only:
            status, reason = "ignored", "c_closed_form_only skips C-table construction"
        elif name == "propagators" and jobs != 1:
            reason += "; lazy per-separation table builds are pinned to one process"
        add(f"{name}.n_jobs", jobs, scope, status, reason)
    return out
