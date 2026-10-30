"""What the run would refuse, checked on a parsed config.

:func:`check_run_refusals` makes the refusals that used to come after the
expansion, in the propagator table or in the sweep, so that
``sft-wick validate`` reports them and ``sft-wick run`` stops before any
stage.  It reuses the code the run refuses with wherever that code can be
reached without a table.  :func:`check_after_expansion` makes the refusals
only the diagrams can decide.
"""

from __future__ import annotations

import os
import stat
from typing import Any

import numpy as np

from . import config as cf
from ._messages import shown
from .config import ConfigError, WorkflowConfig, _refused_at


#: Nodes per axis a cubic ``RectBivariateSpline`` needs, for a lazy C table.
_LAZY_TIME_NODES = 4
#: Nodes per axis each ``interp_method`` needs on a full C table.
_TABLE_NODES = {"slinear": 2, "cubic": 4, "pchip": 4, "quintic": 6}


def check_run_refusals(cfg: WorkflowConfig, system) -> None:
    """Every refusal of the run that the config and ``system`` decide."""
    _check_diagonal(cfg, system)
    _check_axes(cfg)
    _check_position_shapes(cfg, system)
    _check_pinned_points(cfg)
    with _refused_at("expand.observable"):
        terms = _order_zero_terms(cfg, system)
    _refuse_joined_points(terms, cfg.sweep)
    _check_outputs(cfg)
    _check_propagators(cfg, system)


def check_after_expansion(cfg: WorkflowConfig, expansion) -> tuple:
    """The selection of the sweep against the diagrams, returned as
    :meth:`~sft_wick.workflow.expansion.Expansion._selection` gives it."""
    sweep = cfg.sweep
    with _refused_at("sweep.vertex_types"):
        selection = expansion._selection(sweep.orders, sweep.vertex_types, "sweep")
    _refuse_joined_points([task[3] for task in selection[2]], sweep)
    return selection


def _check_diagonal(cfg: WorkflowConfig, system) -> None:
    """The refusals of :meth:`System.expand` and :meth:`System.propagators`
    of ``diag_R`` and ``diag_C``, at the key that states each."""
    from .system import _DIAG_C_REFUSAL, _DIAG_R_REFUSAL

    expand = cfg.expand
    iso_R = system.iso_R if expand.iso_R is None else expand.iso_R
    if expand.diag_R and not iso_R and system._r_offdiagonal():
        raise ConfigError("invalid_value", "expand.diag_R", _DIAG_R_REFUSAL)
    if not (cfg.propagators.diag_C and system._c_offdiagonal()):
        return
    where = "expand.diag_C" if expand.diag_C else "propagators.diag_C"
    raise ConfigError("invalid_value", where, _DIAG_C_REFUSAL)


def _check_axes(cfg: WorkflowConfig) -> None:
    """An empty axis or a repeated value, which :meth:`Expansion.sweep`
    refuses once the expansion and the table are done."""
    from .expansion import _axis_problem

    sweep = cfg.sweep
    axes = {"sweep.t_final_grid": sweep.t_final_grid}
    for block, grid in (
        ("positions_grid", sweep.positions_grid),
        ("external_times_grid", sweep.external_times_grid or {}),
    ):
        axes.update({f"sweep.{block}.{k}": v for k, v in grid.items()})
    for where, values in axes.items():
        problem = _axis_problem(values)
        if problem is not None:
            raise ConfigError("invalid_value", where, problem)


def _selected_orders(cfg: WorkflowConfig):
    return cfg.expand.orders if cfg.sweep.orders is None else cfg.sweep.orders


def _full_table(cfg: WorkflowConfig, system) -> str | None:
    """The homogeneity of the full spatial C table the run builds, or
    ``None`` when it builds a table per separation, or none."""
    props = cfg.propagators
    homogeneity = props.homogeneity or system.homogeneity
    keys = cf._SPATIAL_GRIDS[homogeneity]
    if props.c_closed_form_only or any(getattr(props, k) is None for k in keys):
        return None
    return homogeneity


def _check_position_shapes(cfg: WorkflowConfig, system) -> None:
    """Positions the run cannot combine: on one axis a number next to a
    vector, which ``totals()`` cannot sort, and across points shapes that
    do not broadcast, a vector on a ``general`` full table, and a number
    next to a vector where a ``translation`` full table is read at sampled
    times.  A number next to a vector reaches every other path."""
    shapes: dict = {}
    for point, axis in cfg.sweep.positions_grid.items():
        for i, value in enumerate(axis):
            shape = np.shape(value)
            if bool(shape) != bool(shapes.setdefault(point, shape)):
                raise ConfigError(
                    "invalid_value",
                    f"sweep.positions_grid.{point}[{i}]",
                    "is a number next to a vector, or a vector next to a "
                    "number, on one axis",
                )
    try:
        np.broadcast_shapes(*shapes.values())
    except ValueError:
        found = sorted(set(shapes.values()))
        detail = f"positions of the shapes {found} cannot be subtracted"
        raise ConfigError("invalid_value", "sweep.positions_grid", detail) from None
    vector = next((point for point, shape in shapes.items() if shape), None)
    if vector is None:
        return
    table = _full_table(cfg, system)
    where = f"sweep.positions_grid.{vector}"
    if table == "general":
        detail = "a general full-grid C table supports scalar positions only"
        raise ConfigError("invalid_value", where, detail)
    number_too = any(not shape for shape in shapes.values())
    if table == "translation" and number_too and max(_selected_orders(cfg)) > 0:
        detail = (
            "is a vector next to a number, which a translation full-grid C "
            "table cannot read at an interacting order"
        )
        raise ConfigError("invalid_value", where, detail)


def _check_pinned_points(cfg: WorkflowConfig) -> None:
    """A point cannot be pinned at a time and integrated over time."""
    sweep = cfg.sweep
    pinned = set(sweep.external_times_grid or ())
    names = sweep.integrate_over
    clash = sorted(pinned if names == "all" else pinned & set(names or ()))
    if clash:
        raise ConfigError(
            "invalid_value",
            "sweep.integrate_over",
            f"names {clash}, which sweep.external_times_grid pins; a point "
            f"cannot be pinned at a fixed time and swept at the same time",
        )


def _order_zero_terms(cfg: WorkflowConfig, system) -> list:
    """The diagrams of order 0 the sweep selects, from the Wick contraction
    the expansion would start with.  It holds no vertex, so it costs
    milliseconds, and it knows which pairings vanish (an equal-point R under
    Ito, a causal loop of R)."""
    from sft_wick.perturbation import compute_moment

    from .system import _parse_observable

    expand = cfg.expand
    labels = cfg.sweep.vertex_types
    if 0 not in _selected_orders(cfg) or (labels is not None and "" not in labels):
        return []
    operators, _ = _parse_observable(expand.observable, system, 0)
    result = compute_moment(
        operators,
        system.build_action(),
        order=0,
        response_phase=expand.response_phase,
        ito=expand.ito,
        collect_topology=expand.collect_topology,
        diag_R=expand.diag_R,
        diag_C=cf.effective_expand_diag_C(cfg),
        iso_R=system.iso_R if expand.iso_R is None else expand.iso_R,
        iso_C=expand.iso_C,
        target=expand.target,
    )
    return result.diagram_terms(0)


def _refuse_joined_points(diagram_terms, sweep) -> None:
    """Refuse a grid that puts two points a diagram joins at two positions.

    A chain of response propagators makes its points one spatial point, and
    the integrators refuse them at two positions
    (:func:`~sft_wick.evaluate._require_one_position_per_group`), at the
    grid point that does so.  The grid is a Cartesian product, so the points
    stay together only when each has one position and the two agree.
    """
    grid = sweep.positions_grid
    for term in diagram_terms:
        spatial = term.analyze_spatial()
        for group in spatial.direction_groups:
            points = sorted(
                p for p in group if p in spatial.external_points and p in grid
            )
            for p, q in zip(points, points[1:]):
                if _apart(grid[p], grid[q]):
                    raise ConfigError(
                        "invalid_value",
                        f"sweep.positions_grid.{q}",
                        f"the points {shown(p)} and {shown(q)} take different positions, "
                        f"and a chain of response propagators joins them in a "
                        f"diagram, so they are one spatial point; give both "
                        f"one position",
                    )


def _apart(first: list, second: list) -> bool:
    """Whether two position axes hold a pair of different positions."""
    return (
        len(first) != 1
        or len(second) != 1
        or not np.array_equal(np.asarray(first[0]), np.asarray(second[0]))
    )


def _sweep_columns(cfg: WorkflowConfig) -> tuple:
    """The columns of a sweep row, which is what a ``plot`` output reads."""
    from .expansion import _DIAGRAM_COLUMNS, component_columns

    sweep = cfg.sweep
    return (
        *sweep.positions_grid,
        "t_final",
        *(f"t_{p}" for p in sweep.external_times_grid or ()),
        *component_columns(len(cfg.expand.observable)),
        *_DIAGRAM_COLUMNS,
    )


def _check_outputs(cfg: WorkflowConfig) -> None:
    """What the outputs would raise after the sweep."""
    columns = _sweep_columns(cfg)
    for key, path in (
        ("expand.cache_path", cfg.expand.cache_path),
        ("propagators.cache_path", cfg.propagators.cache_path),
    ):
        if path is not None:
            _check_output_path(path, key, directory=True)
    for i, out in enumerate(cfg.output):
        where = f"output[{i}]"
        if out.path is not None:
            _check_output_path(_written_file(out), f"{where}.path")
        if out.type == "plot":
            _check_plot_format(out.path, f"{where}.path")
            _check_column(out.x, f"{where}.x", columns)
            _check_column(out.y, f"{where}.y", columns)
            if out.hue is not None:
                _check_column(out.hue, f"{where}.hue", columns)
            if out.facet_col is not None:
                _check_column(out.facet_col, f"{where}.facet_col", columns)


def _written_file(out) -> str:
    """The file an output writes: ``np.savez`` adds ``.npz``, and ``savefig``
    the default format to a name with no suffix."""
    path = os.fspath(out.path)
    if out.type == "npz" and not path.endswith(".npz"):
        return f"{path}.npz"
    if out.type == "plot" and not os.path.splitext(path)[1][1:]:
        import matplotlib

        return f"{path.rstrip('.')}.{matplotlib.rcParams['savefig.format']}"
    return path


def _check_plot_format(path: Any, where: str) -> None:
    """A suffix ``savefig`` has no format for, which it raises for once the
    sweep is done."""
    from matplotlib.backend_bases import FigureCanvasBase

    suffix = os.path.splitext(os.fspath(path))[1][1:].lower()
    formats = FigureCanvasBase.get_supported_filetypes()
    if suffix and suffix not in formats:
        detail = f"has the suffix {shown(suffix)}; the formats are {sorted(formats)}"
        raise ConfigError("invalid_value", where, detail, cf._nearest(suffix, formats))


def _check_column(name: Any, where: str, columns: tuple) -> None:
    if not (isinstance(name, str) and name in columns):
        raise ConfigError(
            "invalid_value",
            where,
            f"must be a column of the sweep, one of {cf._or_list(columns)}; "
            f"got {shown(name)}",
            cf._nearest(name, columns) if isinstance(name, str) else (),
        )


def _check_output_path(path: Any, where: str, directory: bool = False) -> None:
    """A path the run could not write to, as far as it can be told without
    writing: a name too long, a parent that is a file, and a directory
    unless ``directory`` allows one (a cache path may be)."""
    try:
        mode = os.stat(path).st_mode
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ConfigError("invalid_value", where, f"cannot be written: {exc}") from None
    if stat.S_ISDIR(mode) and not directory:
        raise ConfigError("invalid_value", where, f"{shown(str(path))} is a directory")


def _check_propagators(cfg: WorkflowConfig, system) -> None:
    """What :meth:`System.propagators` refuses before it builds a table,
    with the closed-form C module imported as the run imports it."""
    from .propagators import _resolve_closed_form_flags

    props = cfg.propagators
    with _refused_at("propagators.t_max"):
        system._require_gamma_grid(props.t_max)
    c_fn = cf._load_c_closed_form(props)
    only, vectorized = props.c_closed_form_only, props.c_closed_form_vectorized
    flag = "vectorized" if vectorized and not only else "only"
    with _refused_at(f"propagators.c_closed_form_{flag}"):
        _resolve_closed_form_flags(system, c_fn, only, vectorized)
    if not only:
        _check_table(cfg, system)


def _check_table(cfg: WorkflowConfig, system) -> None:
    """What a C table needs of its range and of its nodes."""
    props = cfg.propagators
    if props.t_max <= system.t_min:
        raise ConfigError(
            "invalid_value",
            "propagators.t_max",
            f"must be above system.t_min = {system.t_min}; got {props.t_max}",
        )
    table = _full_table(cfg, system)
    if table is None:
        if props.n_grid_t < _LAZY_TIME_NODES:
            raise ConfigError(
                "invalid_value",
                "propagators.n_grid_t",
                f"must be at least {_LAZY_TIME_NODES} when no spatial grid is "
                f"given: the table of each separation is a cubic spline in "
                f"time; got {props.n_grid_t}",
            )
        return
    need = _TABLE_NODES.get(props.interp_method, 1)
    axes = ("n_grid_t", *(k for k in cf._SPATIAL_GRIDS[table] if k.startswith("n_")))
    for key in axes:
        if getattr(props, key) < need:
            raise ConfigError(
                "invalid_value",
                f"propagators.{key}",
                f"must be at least {need} for interp_method "
                f"{shown(props.interp_method)}; got {getattr(props, key)}",
            )
