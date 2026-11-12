"""What the run would refuse, checked on a parsed config.

:func:`check_run_refusals` makes the refusals that used to come after the
expansion, in the propagator table or in the sweep, so that
``sft-wick validate`` reports them and ``sft-wick run`` stops before any
stage.  It reuses the code the run refuses with wherever that code can be
reached without a table.  :func:`check_after_expansion` makes the refusals
only the diagrams can decide.
"""

from __future__ import annotations

import itertools
import os
import stat
from typing import Any, Iterator

import numpy as np

from sft_wick.evaluate import _outside_table, _separation

from . import config as cf
from ._messages import shown
from .config import ConfigError, WorkflowConfig, _refused_at


#: Nodes per axis a cubic ``RectBivariateSpline`` needs, for a lazy C table.
_LAZY_TIME_NODES = 4
#: Nodes per axis each ``interp_method`` needs on a full C table.
_TABLE_NODES = {"slinear": 2, "cubic": 4, "pchip": 4, "quintic": 6}


def check_run_refusals(cfg: WorkflowConfig, system) -> None:
    """Every refusal of the run that the config and ``system`` decide."""
    _check_topology(cfg)
    _check_diagonal(cfg, system)
    _check_axes(cfg)
    _check_isotropy(cfg, system)
    _check_position_shapes(cfg, system)
    _check_diagonal_at_positions(cfg, system)
    _check_pinned_points(cfg)
    with _refused_at("expand.observable"):
        _refuse_joined_points(
            itertools.chain(
                _order_zero_terms(cfg, system), _interacting_terms(cfg, system)
            ),
            cfg.sweep,
        )
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


def _check_topology(cfg: WorkflowConfig) -> None:
    """The refusal of :meth:`System.expand` of ``collect_topology=False``."""
    from .system import _COLLECT_TOPOLOGY_REFUSAL

    if not cfg.expand.collect_topology and max(cfg.expand.orders, default=0) > 0:
        raise ConfigError(
            "invalid_value", "expand.collect_topology", _COLLECT_TOPOLOGY_REFUSAL
        )


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


def _check_diagonal_at_positions(cfg: WorkflowConfig, system) -> None:
    """The refusal of :meth:`Expansion.sweep` of ``diag_C``, which reads a
    callable κ² or σ² at the positions and up to the latest time of the
    grid.  ``propagators.diag_C: false`` turns ``expand.diag_C`` off too, so
    nothing is read then."""
    from .expansion import _position_pairs
    from .system import _diag_c_refusal_at

    if not cfg.propagators.diag_C or not system._position_kernels():
        return
    grid = cfg.sweep.positions_grid
    hit = system._noise_offdiagonal_at(
        _position_pairs(
            [grid.get(p, [0.0]) for p in sorted(cf._observable_points(cfg.expand))]
        ),
        max((t for _path, t in _read_times(cfg)), default=float(system.t_min)),
    )
    if hit is None:
        return
    on_expansion = cfg.expand.diag_C
    detail = _diag_c_refusal_at(
        hit,
        "the sweep",
        (
            "set in System.expand and set on the propagator tables"
            if on_expansion
            else "set on the propagator tables"
        ),
    )
    where = "expand.diag_C" if on_expansion else "propagators.diag_C"
    raise ConfigError("invalid_value", where, detail)


def _check_isotropy(cfg: WorkflowConfig, system) -> None:
    """The refusals of the integrators of ``iso_R`` and ``iso_C`` that the
    config decides, at the key that states each: an index-free R on a matrix
    R, and an index-free C on a C that is no multiple of the identity.

    C is read at the first grid point and the last ``t_final``, by a rule of
    few nodes: rates that differ show there, and two equal entries take equal
    values under any rule.  A user closed form replaces C and is read only by
    the run.  So is a target other than ``moment``, whose diagrams depend on
    the C pairings too, and a kernel that raises at that point.
    """
    from sft_wick.evaluate import _R_MISMATCH

    expand = cfg.expand
    wanted = expand.iso_R and not system.iso_R, expand.iso_C
    if system.n_components == 1 or expand.target != "moment" or not any(wanted):
        return
    reads_r, reads_c = _reads_r_and_c(cfg, system)
    if wanted[0] and reads_r:
        detail = _R_MISMATCH["matrix"].format(r="An index-free R")
        raise ConfigError("invalid_value", "expand.iso_R", detail)
    if wanted[1] and reads_c and cfg.propagators.c_closed_form_module is None:
        detail = _isotropy_refusal(cfg, system)
        if detail is not None:
            raise ConfigError("invalid_value", "expand.iso_C", detail)


def _read_propagators(seq) -> Iterator[list]:
    """Per R class of ``seq``, what the integrators read: an R absorbed into
    an ``already_R_contracted`` vertex carries no factor."""
    from sft_wick.wick import r_class_pairings

    absorbed = {
        leg
        for vi in seq.instances
        if vi.vertex.already_R_contracted
        for leg in vi.spatial_variables
    }
    for props, _ in r_class_pairings(seq.operators):
        yield [
            p for p in props if not (p.kind == "R" and p.spatial_right in absorbed)
        ]


def _reads_r_and_c(cfg: WorkflowConfig, system) -> tuple[bool, bool]:
    """Whether a diagram the sweep selects reads an R, and whether one reads
    a C."""
    from .system import _parse_observable

    operators, _ = _parse_observable(cfg.expand.observable, system, 0)
    pairings = itertools.chain(
        (term.propagators for term in _order_zero_terms(cfg, system)),
        (
            props
            for seq in _selected_sequences(cfg, operators, system.build_action())
            for props in _read_propagators(seq)
        ),
    )
    reads_r = reads_c = False
    for props in pairings:
        kinds = {p.kind for p in props}
        reads_r, reads_c = reads_r or "R" in kinds, reads_c or "C" in kinds
        if reads_r and reads_c:
            break
    return reads_r, reads_c


def _isotropy_refusal(cfg: WorkflowConfig, system) -> str | None:
    """What ``_isotropic_C`` says of C at one point of the sweep, or ``None``
    when it takes that C or the kernel cannot be read there."""
    from sft_wick.evaluate import _C_value_direct_gl, _isotropic_C

    points = [axis[0] for axis in cfg.sweep.positions_grid.values()]
    t = max(cfg.sweep.t_final_grid)
    model = system.build_propagator_model(diag_C=cfg.propagators.diag_C)
    try:
        value = _C_value_direct_gl(model, points[0], t, points[-1], t, n_gauss=4)
    except Exception:  # noqa: BLE001 (a user callable)
        return None
    try:
        _isotropic_C(value)
    except ValueError as exc:
        return str(exc)
    return None


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
    axis_shapes: dict = {}
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
            axis_shapes.setdefault(point, {}).setdefault(shape, i)
    try:
        np.broadcast_shapes(*shapes.values())
    except ValueError:
        found = sorted(set(shapes.values()))
        detail = f"positions of the shapes {found} cannot be subtracted"
        raise ConfigError("invalid_value", "sweep.positions_grid", detail) from None
    # Alternatives on one axis never coexist.  Compare every distinct shape
    # across different axes, without expanding the Cartesian position grid.
    for (p, p_shapes), (q, q_shapes) in itertools.combinations(axis_shapes.items(), 2):
        for (p_shape, i), (q_shape, j) in itertools.product(
            p_shapes.items(), q_shapes.items()
        ):
            if i == 0 and j == 0:
                continue  # The first shapes were checked above.
            try:
                np.broadcast_shapes(p_shape, q_shape)
            except ValueError:
                point, index = (q, j) if j else (p, i)
                detail = (
                    f"positions {shown(p)}[{i}] and {shown(q)}[{j}] of shapes "
                    f"{p_shape} and {q_shape} cannot be subtracted"
                )
                raise ConfigError(
                    "invalid_value", f"sweep.positions_grid.{point}[{index}]", detail
                ) from None
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


def _order_zero_terms(cfg: WorkflowConfig, system) -> Iterator:
    """One diagram of order 0 per R assignment the sweep selects.

    :func:`~sft_wick.wick.r_class_pairings` knows which assignments vanish,
    and the checks read nothing from a diagram but its R.  The connected
    target keeps the one pairing of a two-point observable and no pairing
    above two points, which leave too few edges to connect them.
    """
    from sft_wick._connected import connected_sequence_possible, validate_target
    from sft_wick.perturbation import (
        _order_zero_term,
        check_literal_component_labels,
    )
    from sft_wick.wick import r_class_pairings

    from .system import _parse_observable

    expand = cfg.expand
    labels = cfg.sweep.vertex_types
    if 0 not in _selected_orders(cfg) or (labels is not None and "" not in labels):
        return
    operators, _ = _parse_observable(expand.observable, system, 0)
    validate_target(expand.target, operators)
    check_literal_component_labels(operators)
    if expand.target == "connected" and not connected_sequence_possible(
        len(operators), ()
    ):
        return
    n_response = sum(op.is_response for op in operators)
    for props, _ in r_class_pairings(operators):
        yield _order_zero_term(props, n_response)


def _interacting_terms(cfg: WorkflowConfig, system) -> Iterator:
    """One diagram per R assignment at the selected orders above 0, as the
    walk reaches it, so a refusal stops the walk at its first diagram.

    The walk goes over R assignments only, so its cost does not grow with
    the C pairings of the full expansion.  It is skipped when no two points
    of the grid are apart, and when nothing can join two external points:
    that takes a response leg in the observable or a local vertex with two
    or more of them.  A connected target decides by its C pairings as well,
    so :func:`check_after_expansion` makes this check there.
    """
    from sft_wick.perturbation import _order_zero_term
    from sft_wick.wick import r_class_pairings

    from .system import _parse_observable

    expand = cfg.expand
    grid = cfg.sweep.positions_grid
    if expand.target != "moment" or not any(
        _apart(grid[p], grid[q]) for p, q in itertools.combinations(grid, 2)
    ):
        return
    operators, _ = _parse_observable(expand.observable, system, 0)
    n_response = sum(op.is_response for op in operators)
    action = system.build_action()
    if not n_response and not any(
        v.local and v.n_response >= 2 for v in action.vertices
    ):
        return
    for seq in _selected_sequences(cfg, operators, action):
        for props, _ in r_class_pairings(seq.operators):
            yield _order_zero_term(props, n_response)


def _selected_sequences(cfg: WorkflowConfig, operators, action) -> Iterator:
    """The vertex sequences of the selected orders above 0 and vertex types."""
    from sft_wick.perturbation import _vertex_sequences

    from .expansion import vertex_type_label

    labels = cfg.sweep.vertex_types
    for n in _selected_orders(cfg):
        if n == 0:
            continue
        for seq in _vertex_sequences(list(operators), action, n, "moment"):
            if (
                labels is None
                or vertex_type_label(vi.vertex.coupling for vi in seq.instances)
                in labels
            ):
                yield seq


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
    _check_rate_times(cfg, system)
    c_fn = cf._load_c_closed_form(props)
    only, vectorized = props.c_closed_form_only, props.c_closed_form_vectorized
    flag = "vectorized" if vectorized and not only else "only"
    with _refused_at(f"propagators.c_closed_form_{flag}"):
        _resolve_closed_form_flags(system, c_fn, only, vectorized)
    if not only:
        _check_table(cfg, system)
        _check_table_times(cfg, system)
        _check_table_positions(cfg, system)


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


def _read_times(cfg: WorkflowConfig):
    """``(dotted path, time)`` of each time the sweep reads.

    ``sweep.t_final_grid`` counts only when a point is left free: with every
    external pinned it is not read.
    """
    pinned = cfg.sweep.external_times_grid or {}
    axes = [(f"sweep.external_times_grid.{p}", v) for p, v in pinned.items()]
    if not cf._observable_points(cfg.expand) <= set(pinned):
        axes.insert(0, ("sweep.t_final_grid", cfg.sweep.t_final_grid))
    for axis, values in axes:
        for i, value in enumerate(values):
            yield f"{axis}[{i}]", float(value)


def _refuse_time_outside(cfg, lo, hi, lo_key, hi_key, reason) -> None:
    """Refuse a time the sweep reads outside ``[lo, hi]``."""
    for where, t in _read_times(cfg):
        if _outside_table(t, lo, hi):
            limit = f"{hi_key} = {hi}" if t > hi else f"{lo_key} = {lo}"
            raise ConfigError(
                "invalid_value", where, f"time {shown(t)} is past {limit}, {reason}"
            )


def _check_table_times(cfg: WorkflowConfig, system) -> None:
    """A time outside the C table, which would be clamped or extrapolated;
    see :func:`~sft_wick.workflow.expansion._guard_table_reach`."""
    _refuse_time_outside(
        cfg,
        float(system.t_min),
        float(cfg.propagators.t_max),
        "system.t_min",
        "propagators.t_max",
        "where the C table ends; the table would return a clamped or "
        "extrapolated value",
    )


def _check_table_positions(cfg: WorkflowConfig, system) -> None:
    """A position past a full spatial grid, which would be extrapolated.  A
    lazy table builds at any position, and a rotation table's cos is in
    ``[-1, 1]`` by construction."""
    table, grid = _full_table(cfg, system), cfg.sweep.positions_grid
    if table == "translation":
        r_max = float(cfg.propagators.r_max)
        for (p, at_p), (q, at_q) in itertools.combinations(grid.items(), 2):
            for a, (j, b) in itertools.product(at_p, enumerate(at_q)):
                r = _separation(a, b)
                if _outside_table(r, 0.0, r_max):
                    raise ConfigError(
                        "invalid_value",
                        f"sweep.positions_grid.{q}[{j}]",
                        f"{shown(q)} is {shown(r)} from {shown(p)}, past "
                        f"propagators.r_max = {r_max}, where the C table "
                        f"ends; the table would extrapolate.  Raise r_max, or "
                        f"leave r_max and n_grid_r out for a lazy table",
                    )
    if table == "general":
        x_max = float(cfg.propagators.x_max)
        for p, values in grid.items():
            for i, value in enumerate(values):
                if np.any(_outside_table(value, -x_max, x_max)):
                    raise ConfigError(
                        "invalid_value",
                        f"sweep.positions_grid.{p}[{i}]",
                        f"{shown(value)} is past propagators.x_max = {x_max}, "
                        f"where the C table ends; the table would extrapolate."
                        f"  Raise x_max, or leave x_max and n_grid_x out for "
                        f"a lazy table",
                    )


def _check_rate_times(cfg: WorkflowConfig, system) -> None:
    """A time outside the cumulative-rate spline behind R, which matters when
    no C table bounds the time, as under ``c_closed_form_only``.  Reads the
    rule of :func:`~sft_wick.workflow.expansion._request_reads_R`, which the
    run reads too."""
    from .expansion import _request_reads_R
    from .specs import DiagonalA
    from .system import _OBS_PATTERN

    lin = system._effective_linear
    if (
        system.explicit_R is not None
        or not isinstance(lin, DiagonalA)
        or not callable(lin.gamma)
    ):
        return
    names = {_OBS_PATTERN.match(s).group("name") for s in cfg.expand.observable}
    if not _request_reads_R(system, _selected_orders(cfg), names):
        return
    _refuse_time_outside(
        cfg,
        float(lin.t_min_cache),
        float(lin.t_max_cache),
        "system.linear.t_min_cache",
        "system.linear.t_max_cache",
        "where the cumulative-rate spline behind R ends; the spline would "
        "extrapolate",
    )
