"""YAML-driven configuration for the workflow API.

Lets users declare an entire ``System`` / ``Expansion`` /
``Propagators`` / sweep pipeline in a single YAML file, with fields
mapping 1:1 to the Python L1 API so the config is self-documenting.

Invoke via the CLI::

    sft-wick run examples/demo1_config.yaml

Or programmatically::

    from sft_wick.workflow.config import load_workflow_config, run_workflow

    cfg = load_workflow_config("examples/demo1_config.yaml")
    sweep, totals_df = run_workflow(cfg)
"""

from __future__ import annotations

import importlib.util
import math
import os
import re
import sys
from dataclasses import dataclass, field
from dataclasses import fields as _dc_fields
from pathlib import Path
from typing import Any

import numpy as np


# =========================================================================
# Internal dataclasses — a thin typed mirror of the L1 ``System`` spec
# =========================================================================


@dataclass(frozen=True)
class WorkflowConfig:
    """Top-level parsed config."""

    system: "SystemConfig"
    expand: "ExpandConfig"
    propagators: "PropagatorsConfig"
    sweep: "SweepConfig"
    output: list["OutputConfig"] = field(default_factory=list)


@dataclass(frozen=True)
class SystemConfig:
    field_name: str
    n_components: int
    linear: dict  # parsed at build time
    noise: dict
    vertices: list
    nonlocal_vertices: list = field(default_factory=list)
    t_min: float = 0.0
    # Resolved at parse time so that noise.kappa2.type='callable_module'
    # (and any future module-loaded specs in the system block) can resolve
    # paths relative to the YAML file at build time.
    base_dir: Path | None = None


@dataclass(frozen=True)
class ExpandConfig:
    observable: tuple
    orders: tuple
    response_phase: bool = True
    ito: bool = True
    collect_topology: bool = True
    iso_R: Any = None
    diag_R: bool = True
    diag_C: bool = True
    iso_C: bool = False
    cache_path: Any = None
    n_jobs: int = 1


@dataclass(frozen=True)
class PropagatorsConfig:
    t_max: float
    n_grid_t: int = 60
    dt: float | None = None
    homogeneity: Any = None
    r_max: Any = None
    n_grid_r: Any = None
    n_grid_cos: Any = None
    x_max: Any = None
    n_grid_x: Any = None
    n_jobs: int = 1
    #: ``'auto'`` (default): use the built-in closed form when the kernel
    #: family has one; ``null`` / ``false``: always run quadrature.  A
    #: ``c_closed_form_module`` takes precedence over both.
    c_closed_form: Any = "auto"
    c_closed_form_module: Any = None
    c_closed_form_attr: str = "C_fn"
    c_closed_form_only: bool = False
    c_closed_form_vectorized: bool = False
    cache_path: Any = None
    interp_method: str = "linear"
    #: 'auto' (GL with a converged node count for built-in kernels,
    #: dblquad for callable kernels) | 'gauss_legendre' | 'dblquad'
    c_method: str = "auto"
    c_n_gauss: int = 20  # nodes per dim under c_method='gauss_legendre'
    diag_C: bool = True  # set False to preserve off-diagonal C entries
    #                       (e.g. lensing kappa-gamma_+ cross). Requires
    #                       c_closed_form_only=True. When False, also
    #                       sets expand.diag_C=False so the symbolic
    #                       simplification keeps the (a, b) observable
    #                       indices distinct -- without that step the
    #                       order-0 cross pair (a != b) collapses to 0
    #                       via the KroneckerDelta(a, b) inserted by
    #                       DiagramTerm.apply_diagonal.


@dataclass(frozen=True)
class SweepConfig:
    positions_grid: dict
    t_final_grid: list
    #: The component axis: tuples of component indices, one per observable
    #: operator, from YAML ``component_tuples`` or its older 2-point name
    #: ``component_pairs``.  Read it as :attr:`component_tuples`; the field
    #: keeps its old name so existing construction sites keep working.
    component_pairs: list
    orders: Any = None
    vertex_types: Any = None
    integrate_over: Any = None
    method: str = "qmc_vectorized"
    n_samples: int = 2 ** 13
    #: Sobol seed; ``None`` (YAML ``null``) draws an unseeded sequence.
    seed: int | None = 42
    n_jobs: int = 1
    n_gauss: int = 8  # used only when method='gauss_legendre'
    #: ``{point: [times]}`` pinning externals at UNEQUAL times, swept as a
    #: further Cartesian axis (mirrors ``positions_grid``).  Omit to pin every
    #: external at ``t_final``, which is what every sweep did before -- and
    #: which makes any observable with a response leg identically 0, since
    #: Theta kills the R joining two externals at the same time.
    external_times_grid: dict | None = None

    @property
    def component_tuples(self) -> list:
        """The component axis (one index per observable operator)."""
        return self.component_pairs


@dataclass(frozen=True)
class OutputConfig:
    type: str  # "table" | "npz" | "plot"
    path: Any = None
    # Type-specific:
    format: str = "markdown"   # for table
    x: Any = None              # for plot
    y: str = "value"
    hue: Any = "order"
    facet_col: Any = None


# =========================================================================
# YAML → WorkflowConfig
# =========================================================================


def load_workflow_config(
    path: str | Path,
    overrides: dict | None = None,
) -> WorkflowConfig:
    """Load and validate a workflow YAML config.

    Args:
        path: path to a YAML file.
        overrides: optional ``{dotted.key: value}`` dict to patch
            the loaded config (e.g. ``{"sweep.seed": 7}``).

    Returns:
        A :class:`WorkflowConfig` ready to pass to :func:`run_workflow`.
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "YAML workflow configs require PyYAML.  "
            "Install with `pip install pyyaml`."
        ) from e

    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f)

    if overrides:
        for dotted_key, value in overrides.items():
            _apply_override(data, dotted_key, value)

    return _parse_workflow(data, base_dir=path.parent)


_OVERRIDE_SEGMENT = re.compile(r"^(?P<name>[^.\[\]]+)(?P<idx>(?:\[\d+\])*)$")


def _override_path(dotted_key: str) -> list:
    """``"output[0].path"`` -> ``["output", 0, "path"]``.

    List indices are part of the documented ``--override`` syntax
    (``--override "output[0].path=results.md"``); before, the whole
    segment was looked up as a dict key and every such override raised.
    """
    path: list = []
    for part in dotted_key.split("."):
        m = _OVERRIDE_SEGMENT.match(part)
        if m is None:
            raise KeyError(
                f"override key '{dotted_key}': cannot read the segment "
                f"'{part}'; write name, or name[index] for a list entry."
            )
        path.append(m.group("name"))
        path.extend(int(i) for i in re.findall(r"\[(\d+)\]", m.group("idx")))
    return path


def _override_step(cur: Any, step: Any, dotted_key: str):
    """One step along an override path; raises if it does not exist."""
    if isinstance(step, int):
        if not isinstance(cur, list) or not 0 <= step < len(cur):
            raise KeyError(
                f"override key '{dotted_key}' does not exist in config"
            )
    elif not isinstance(cur, dict) or step not in cur:
        raise KeyError(
            f"override key '{dotted_key}' does not exist in config"
        )
    return cur[step]


def _apply_override(data: dict, dotted_key: str, value: Any) -> None:
    """Apply an ``"a.b.c"`` (or ``"a[0].b"``) override to a parsed config."""
    path = _override_path(dotted_key)
    cur: Any = data
    for step in path[:-1]:
        cur = _override_step(cur, step, dotted_key)
    leaf = path[-1]
    _override_step(cur, leaf, dotted_key)      # must already exist
    cur[leaf] = value


def _parse_workflow(data: dict, base_dir: Path) -> WorkflowConfig:
    system_d = _require_dict(data, "system")
    expand_d = _require_dict(data, "expand")
    props_d = _require_dict(data, "propagators")
    sweep_d = _require_dict(data, "sweep")
    output_d = data.get("output", [])

    # Extract the top-level dt before parsing system so the linear gamma-spline
    # cache can derive its own n_grid_cache from the same dt by default.
    default_dt = props_d.get("dt")
    if default_dt is not None:
        default_dt = float(default_dt)

    system_cfg = _parse_system(system_d, base_dir, default_dt=default_dt,
                               t_max=props_d.get("t_max"))
    expand_cfg = _parse_expand(expand_d)
    props_cfg = _parse_propagators(props_d, base_dir)
    sweep_cfg = _parse_sweep(sweep_d)
    output_cfgs = [_parse_output(o) for o in (output_d or [])]

    return WorkflowConfig(
        system=system_cfg,
        expand=expand_cfg,
        propagators=props_cfg,
        sweep=sweep_cfg,
        output=output_cfgs,
    )


def _require_dict(d: dict, key: str) -> dict:
    if key not in d:
        raise ValueError(f"config missing required top-level section '{key}'")
    if not isinstance(d[key], dict):
        raise ValueError(f"config section '{key}' must be a mapping")
    return d[key]


def _parse_system(
    d: dict, base_dir: Path, *, default_dt: float | None = None,
    t_max: Any = None,
) -> SystemConfig:
    fld = d.get("field", {}) or {}
    name = fld.get("name", "phi")
    nc = int(fld.get("n_components", 1))
    t_min = float(d.get("t_min", 0.0))

    linear = d.get("linear")
    if linear is None:
        raise ValueError("system.linear is required")
    if not isinstance(linear, dict):
        raise ValueError(f"system.linear must be a mapping; got {linear!r}.")
    linear = _resolve_linear(dict(linear), base_dir, default_dt=default_dt,
                             n_components=nc, t_min=t_min, t_max=t_max)

    noise = d.get("noise")
    if noise is None:
        raise ValueError("system.noise is required")

    # Validate each vertex block and resolve its coupling (file paths are
    # relative to the YAML file).
    vertices = [
        _parse_vertex(v, base_dir, kind="local", index=i, n_components=nc)
        for i, v in enumerate(d.get("vertices", []) or [])
    ]
    nonlocal_vertices = [
        _parse_vertex(v, base_dir, kind="nonlocal", index=i, n_components=nc)
        for i, v in enumerate(d.get("nonlocal_vertices", []) or [])
    ]
    names = [v["name"] for v in vertices + nonlocal_vertices]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(
            f"system: vertex name(s) {repeated} used more than once.  The "
            f"coupling values are keyed by vertex name, so names must be "
            f"unique across vertices and nonlocal_vertices."
        )

    return SystemConfig(
        field_name=name, n_components=nc,
        linear=linear, noise=noise,
        vertices=vertices, nonlocal_vertices=nonlocal_vertices,
        t_min=t_min,
        base_dir=base_dir,
    )


#: The three ways a vertex block can give its coupling.
_COUPLING_SOURCES = ("coupling", "coupling_path", "coupling_module")
_VERTEX_FLAGS = ("coupling_vectorized", "equal_time", "already_R_contracted")


def _vertex_keys(cls) -> set:
    """The keys of a vertex block: every field of ``cls`` -- so a field
    added to :class:`LocalVertex` / :class:`NonLocalVertex` is accepted
    without a change here -- plus the coupling sources."""
    return ({f.name for f in _dc_fields(cls)} | set(_COUPLING_SOURCES)
            | {"coupling_attr"})


def _parse_vertex(v: Any, base_dir: Path, *, kind: str, index: int,
                  n_components: int) -> dict:
    """Validate one ``system.vertices[]`` (``kind='local'``) or
    ``system.nonlocal_vertices[]`` (``kind='nonlocal'``) entry and resolve
    its coupling to an array or a callable.

    Exactly one coupling source:
      ``coupling``          — inline tensor (nested YAML lists).
      ``coupling_path``     — path to an ``.npy`` file, loaded as a
                              numpy array.
      ``coupling_module``   — path to a ``.py`` module exporting an
                              attribute (``coupling_attr``, default
                              ``coupling_fn``) used as a callable
                              ``fn(n_list, t_list) -> tensor``.
                              Required for spacetime-dependent
                              non-local vertices like demo2's ``κ^{(3)}``.

    A tensor coupling must have every axis of length ``N``: shape
    ``(N,)*order`` for a non-local vertex, ``(N,)*n`` (``n >= 1``, the
    first axis the ψ leg) for a local one.  A tensor of another shape used
    to be indexed without an error.
    """
    from . import specs as sp

    block = "vertices" if kind == "local" else "nonlocal_vertices"
    cls = sp.LocalVertex if kind == "local" else sp.NonLocalVertex
    where = f"system.{block}[{index}]"
    if not isinstance(v, dict):
        raise ValueError(
            f"{where} must be a mapping with 'name' and a coupling; "
            f"got {v!r}."
        )
    _reject_unknown_keys(v, _vertex_keys(cls), where)
    out = dict(v)
    name = out.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{where}: 'name' (a non-empty string) is required.")
    where = f"{where} ({name!r})"
    sources = [k for k in _COUPLING_SOURCES if k in out]
    if len(sources) != 1:
        raise ValueError(
            f"{where}: give exactly one of {list(_COUPLING_SOURCES)}; "
            f"got {sources or 'none'}."
        )
    if "coupling_attr" in out and sources[0] != "coupling_module":
        raise ValueError(
            f"{where}: 'coupling_attr' names the callable in "
            f"'coupling_module' and does nothing with {sources[0]!r}."
        )
    if kind == "nonlocal":
        order = out.get("order")
        if (isinstance(order, bool) or not isinstance(order, (int, np.integer))
                or order < 1):
            raise ValueError(
                f"{where}: 'order' (m, the number of psi legs, an integer "
                f">= 1) is required; got {order!r}."
            )
        out["order"] = int(order)
    for flag in _VERTEX_FLAGS:
        if flag in out:
            out[flag] = _as_bool(out[flag], f"{where}.{flag}")

    if sources[0] == "coupling_path":
        out["coupling"] = np.load(
            (base_dir / str(out.pop("coupling_path"))).resolve())
    elif sources[0] == "coupling_module":
        mod_path = (base_dir / str(out.pop("coupling_module"))).resolve()
        attr = str(out.pop("coupling_attr", "coupling_fn"))
        out["coupling"] = _load_callable_from_module(mod_path, attr)
    if callable(out["coupling"]):
        return out

    if out.get("coupling_vectorized"):
        raise ValueError(
            f"{where}: coupling_vectorized applies to a callable coupling "
            f"(coupling_module), not to a tensor."
        )
    try:
        arr = np.asarray(out["coupling"])
    except ValueError:
        arr = np.asarray(None)
    if not (np.issubdtype(arr.dtype, np.number)
            and np.all(np.isfinite(arr))):
        raise ValueError(
            f"{where}: the coupling must be a tensor of finite numbers."
        )
    n = int(n_components)
    if kind == "nonlocal":
        want = (n,) * out["order"]
    else:
        if arr.ndim == 0:
            raise ValueError(
                f"{where}: a local coupling F^(n) is a tensor with n >= 1 "
                f"axes, the first the psi leg; got a scalar."
            )
        want = (n,) * arr.ndim
    if arr.shape != want:
        raise ValueError(
            f"{where}: the coupling has shape {arr.shape}; with "
            f"n_components = {n} it must be {want}."
        )
    out["coupling"] = arr
    return out


def _build_vertex(cls, v: dict):
    """A :class:`LocalVertex` / :class:`NonLocalVertex` from a parsed vertex
    block.  Every dataclass field the block names is passed on, so no field
    of the class is out of reach from YAML."""
    c = v["coupling"]
    kwargs = {"coupling": c if callable(c) else np.asarray(c)}
    for f in _dc_fields(cls):
        if f.name != "coupling" and f.name in v:
            kwargs[f.name] = v[f.name]
    if "order" in kwargs:
        kwargs["order"] = int(kwargs["order"])
    for flag in _VERTEX_FLAGS:
        if flag in kwargs:
            kwargs[flag] = _as_bool(kwargs[flag], flag)
    return cls(**kwargs)


#: Keys of each ``system.linear`` type.
_LINEAR_DIAGONAL_KEYS = ("type", "gamma", "gamma_module", "gamma_attr", "dt",
                         "n_grid_cache", "t_max_cache", "t_min_cache")
_LINEAR_EXPLICIT_KEYS = ("type", "R_time_module", "R_time_attr", "iso_R")


def _resolve_linear(
    lin: dict, base_dir: Path, *, default_dt: float | None = None,
    n_components: int | None = None, t_min: float = 0.0, t_max: Any = None,
) -> dict:
    """Resolve ``system.linear`` based on its ``type`` field.

    Supported lowerings:

    ``type: diagonal`` (default) -> :class:`sft_wick.workflow.specs.DiagonalA`

        ``gamma``: inline list of floats, or a 1D nested-list array.
        ``gamma_module``: path to a ``.py`` module exporting an attribute
            (default ``gamma``) used as a callable ``gamma(t) -> array(N)``.
            Required for spacetime-dependent linear drift such as the
            Sachs-saddle ``2 theta^(sa)(lambda)``.

        Discretization: the spline cache uses ``n_grid_cache`` points
        uniformly on ``[t_min_cache, t_max_cache]`` (defaults 0 and
        100). When the user provides ``dt`` (here or via
        ``propagators.dt``), it is converted to
        ``n_grid_cache = ceil((t_max_cache - t_min_cache) / dt)`` so a
        single ``dt`` controls every grid in the workflow. Providing both
        ``dt`` and ``n_grid_cache`` is rejected to avoid ambiguity.

    ``type: explicit`` -> :class:`sft_wick.workflow.specs.ExplicitR`

        Escape hatch for a closed-form R: the user supplies
        ``R(t1, t2)`` directly, so the wrapper bypasses the
        gamma-spline cache entirely. This unlocks YAML use cases the
        diagonal lowering can't express -- e.g. a dense drift matrix,
        causal kernels with non-exponential decay, or pre-computed
        spline callables loaded from disk.

        ``R_time_module``: path to a ``.py`` module exporting an
            attribute (default ``R_time``) used as a callable
            ``R_time(t1, t2) -> float | (N, N)``. Must enforce
            causality (return 0 when ``t1 < t2``).
        ``iso_R``: ``true`` (default) for a scalar R, ``false`` for an
            ``(N, N)`` matrix.  The callable is called once at a causal
            and once at an acausal time pair inside ``[t_min, t_max]``:
            the value must have the declared shape and vanish for
            ``t1 < t2``.  A matrix R with off-diagonal entries also
            needs ``expand.diag_R: false`` and ``expand.diag_C: false``
            (``System.expand`` refuses it otherwise).

        γ-spline cache knobs (``gamma``, ``gamma_module``, ``dt``,
        ``n_grid_cache``, ``t_max_cache``, ``t_min_cache``) do not apply
        under this type and raise if specified -- the propagator is the
        user's callable, not a derived spline.
    """
    lt = lin.get("type", "diagonal")
    if lt == "explicit":
        return _resolve_linear_explicit(lin, base_dir,
                                        n_components=n_components,
                                        t_min=t_min, t_max=t_max)
    if lt == "diagonal":
        return _resolve_linear_diagonal(lin, base_dir, default_dt=default_dt,
                                        n_components=n_components)
    raise ValueError(
        f"Unsupported linear operator type {lt!r}.  "
        f"Supported: 'diagonal', 'explicit'."
    )


def _decay_rates(gamma: Any, n_components: int | None) -> list:
    """``system.linear.gamma``: one decay rate per component, or one for
    all (a single number, as ``--override system.linear.gamma=1.5``
    writes it, counts as one for all)."""
    where = "system.linear.gamma"
    if not isinstance(gamma, list):
        return [_yaml_number(gamma, where)]
    if not gamma:
        raise ValueError(f"{where} is an empty list.")
    rates = [_yaml_number(g, f"{where}[{i}]") for i, g in enumerate(gamma)]
    if n_components is not None and len(rates) not in (1, int(n_components)):
        raise ValueError(
            f"{where} has {len(rates)} rates; with n_components = "
            f"{n_components} give {n_components}, or one for all."
        )
    return rates


def _resolve_linear_diagonal(
    lin: dict, base_dir: Path, *, default_dt: float | None = None,
    n_components: int | None = None,
) -> dict:
    """Parse-time resolver for ``type: diagonal``."""
    _reject_unknown_keys(lin, _LINEAR_DIAGONAL_KEYS,
                         "system.linear (type 'diagonal')")
    if "gamma" in lin and "gamma_module" in lin:
        raise ValueError(
            "system.linear: provide exactly one of {'gamma', 'gamma_module'}"
        )
    if "gamma_module" in lin:
        mod_path = (base_dir / lin.pop("gamma_module")).resolve()
        attr = lin.pop("gamma_attr", "gamma")
        lin["gamma"] = _load_callable_from_module(mod_path, attr)
    elif "gamma" not in lin:
        raise ValueError(
            "system.linear (type 'diagonal') requires 'gamma', a list of "
            "decay rates (one per component, or one for all), or "
            "'gamma_module' for a time-dependent rate."
        )
    elif "gamma_attr" in lin:
        raise ValueError(
            "system.linear: 'gamma_attr' names the callable in "
            "'gamma_module' and does nothing with 'gamma'."
        )
    else:
        lin["gamma"] = _decay_rates(lin["gamma"], n_components)

    # dt -> n_grid_cache derivation. linear.dt overrides propagators.dt.
    linear_dt = lin.pop("dt", None)
    effective_dt = float(linear_dt) if linear_dt is not None else default_dt
    if effective_dt is not None:
        if effective_dt <= 0.0:
            raise ValueError(
                f"system.linear.dt (or propagators.dt) must be positive, "
                f"got {effective_dt}"
            )
        if "n_grid_cache" in lin:
            raise ValueError(
                "system.linear: specify exactly one of {'dt' (here or in "
                "propagators), 'n_grid_cache'}; got both."
            )
        t_max_cache = float(lin.get("t_max_cache", 100.0))
        t_min_cache = float(lin.get("t_min_cache", 0.0))
        lin["n_grid_cache"] = max(
            2, int(math.ceil((t_max_cache - t_min_cache) / effective_dt)))
    return lin


def _resolve_linear_explicit(
    lin: dict, base_dir: Path, *, n_components: int | None = None,
    t_min: float = 0.0, t_max: Any = None,
) -> dict:
    """Parse-time resolver for ``type: explicit``.

    Loads ``R_time`` from a user module and rejects fields that only
    make sense under the diagonal-with-gamma-spline path. The module
    is registered for cross-process by-value serialisation by
    :func:`_load_callable_from_module`, so the loaded callable composes
    with ``propagators.n_jobs > 1`` / ``sweep.n_jobs > 1`` even when
    joblib reuses a worker pool across calls.

    ``iso_R: false`` declares an ``(N, N)`` matrix R (a dense drift); the
    callable is probed once for its shape and once for causality.
    """
    forbidden = (
        "gamma", "gamma_module", "gamma_attr",
        "dt", "n_grid_cache", "t_max_cache", "t_min_cache",
    )
    present = [k for k in forbidden if k in lin]
    if present:
        raise ValueError(
            f"system.linear.type='explicit' does not accept "
            f"gamma-spline fields {present!r}.  Use 'type: diagonal' for "
            f"those, or remove them under 'type: explicit'."
        )
    if "R_time" in lin:
        raise ValueError(
            "system.linear.type='explicit' does not support an inline "
            "'R_time' (a callable cannot be expressed in YAML).  Use "
            "'R_time_module' + 'R_time_attr' instead."
        )
    _reject_unknown_keys(lin, _LINEAR_EXPLICIT_KEYS,
                         "system.linear (type 'explicit')")
    if "R_time_module" not in lin:
        raise ValueError(
            "system.linear.type='explicit' requires "
            "'R_time_module: <relative path to .py file>'."
        )
    iso_r = _as_bool(lin.get("iso_R", True), "system.linear.iso_R")
    lin["iso_R"] = iso_r
    mod_path = (base_dir / lin.pop("R_time_module")).resolve()
    attr = lin.pop("R_time_attr", "R_time")
    lin["R_time"] = _load_callable_from_module(mod_path, attr)
    _probe_R_time(lin["R_time"], iso_r, n_components, t_min, t_max)
    return lin


def _probe_R_time(R, iso_R: bool, n_components: int | None,
                  t_min: float, t_max: Any) -> None:
    """Call the user's R once at a causal and once at an acausal time pair
    inside ``[t_min, t_max]``: the value must have the shape ``iso_R``
    declares and must vanish when ``t1 < t2``.  A scalar returned where a
    matrix was declared (or the reverse) surfaced much later and as
    something else; an R written without the Heaviside surfaced not at
    all."""
    where = "system.linear.R_time_module"
    try:
        hi = float(t_max)
    except (TypeError, ValueError):
        hi = float("nan")
    span = (hi - t_min) if (math.isfinite(hi) and hi > t_min) else 1.0
    t1, t2 = t_min + 0.7 * span, t_min + 0.2 * span
    try:
        fwd = np.asarray(R(t1, t2), dtype=float)
        bwd = np.asarray(R(t2, t1), dtype=float)
    except Exception as e:  # noqa: BLE001 -- a user callable
        raise ValueError(
            f"{where}: R_time({t1:g}, {t2:g}) raised "
            f"{type(e).__name__}: {e}"
        ) from e
    if iso_R:
        ok, want = fwd.shape == (), "a scalar"
    elif n_components is None:
        ok = fwd.ndim == 2 and fwd.shape[0] == fwd.shape[1]
        want = "a square matrix"
    else:
        n = int(n_components)
        ok, want = fwd.shape == (n, n), f"an ({n}, {n}) matrix"
    if not ok:
        raise ValueError(
            f"{where}: R_time({t1:g}, {t2:g}) has shape {fwd.shape}, but "
            f"iso_R: {str(iso_R).lower()} declares {want}."
        )
    if not np.all(np.isfinite(fwd)):
        raise ValueError(
            f"{where}: R_time({t1:g}, {t2:g}) is not finite: {fwd.tolist()}."
        )
    if bwd.shape != fwd.shape or np.any(bwd != 0.0):
        raise ValueError(
            f"{where}: R_time({t2:g}, {t1:g}) = {bwd.tolist()}, but the "
            f"response function is causal: R(t1, t2) = 0 for t1 < t2."
        )


def _load_callable_from_module(path: Path, attr: str):
    """Import ``path`` as a standalone module and return
    ``getattr(module, attr)``.

    Registers the module under its file basename (``path.stem``) and ensures
    ``path.parent`` is on both ``sys.path`` (for the current process) and
    the ``PYTHONPATH`` environment variable (for subprocess workers, e.g.
    joblib loky). Without the env-var step, a worker process started after
    this call cannot re-import the module by name, and unpickling a cache
    that holds the loaded callable raises ``BrokenProcessPool`` /
    ``ModuleNotFoundError``.
    """
    parent_dir = str(path.parent.resolve())
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    # Propagate parent_dir to subprocess workers via PYTHONPATH. loky
    # workers inherit os.environ but not the parent's sys.path mods,
    # so this is the durable channel.
    pp = os.environ.get("PYTHONPATH", "")
    pp_parts = pp.split(os.pathsep) if pp else []
    if parent_dir not in pp_parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([parent_dir, *pp_parts])

    module_name = path.stem
    spec_obj = importlib.util.spec_from_file_location(module_name, path)
    if spec_obj is None or spec_obj.loader is None:
        raise ImportError(
            f"Cannot load coupling module {path!r}."
        )
    module = importlib.util.module_from_spec(spec_obj)
    sys.modules[module_name] = module
    spec_obj.loader.exec_module(module)
    fn = getattr(module, attr, None)
    if fn is None or not callable(fn):
        raise AttributeError(
            f"Module {path!r} has no callable attribute {attr!r}."
        )
    _register_module_by_value(module)
    return fn


def _register_module_by_value(module) -> None:
    """Register ``module`` for cross-process by-value serialisation.

    Modules loaded via :func:`importlib.util.spec_from_file_location`
    are not importable by name in subprocess workers. joblib's loky
    backend uses a persistent worker pool whose ``sys.path`` /
    ``PYTHONPATH`` is fixed at first-pool-creation time, so a worker
    spawned during an earlier test won't have a later test's
    ``tmp_path`` available.

    cloudpickle's ``register_pickle_by_value`` flips the encoding so
    the module's source is shipped inline with each task. Workers no
    longer need to import anything by name.

    Falls back silently if cloudpickle is unavailable (joblib pulls
    it in, but a custom install might not).
    """
    try:
        from joblib.externals import cloudpickle
    except ImportError:  # pragma: no cover - joblib pulls in cloudpickle
        return
    try:
        cloudpickle.register_pickle_by_value(module)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        # cloudpickle rejects modules that are part of a package or
        # don't have a real source file. Either case means the worker
        # can already import the module by name, so by-value encoding
        # is not needed.
        pass


def _parse_expand(d: dict) -> ExpandConfig:
    obs = d.get("observable")
    if obs is None:
        raise ValueError("expand.observable is required")
    orders = d.get("orders")
    if orders is None:
        raise ValueError("expand.orders is required")
    return ExpandConfig(
        observable=tuple(obs),
        orders=tuple(int(o) for o in orders),
        response_phase=bool(d.get("response_phase", True)),
        ito=bool(d.get("ito", True)),
        collect_topology=bool(d.get("collect_topology", True)),
        iso_R=d.get("iso_R"),
        diag_R=bool(d.get("diag_R", True)),
        diag_C=bool(d.get("diag_C", True)),
        iso_C=bool(d.get("iso_C", False)),
        cache_path=d.get("cache_path"),
        n_jobs=int(d.get("n_jobs", 1)),
    )


def _parse_propagators(d: dict, base_dir: Path) -> PropagatorsConfig:
    if "t_max" not in d:
        raise ValueError("propagators.t_max is required")
    module_spec = d.get("c_closed_form_module")
    if module_spec is not None:
        module_spec = str((base_dir / module_spec).resolve())

    t_max = float(d["t_max"])
    dt = d.get("dt")
    has_n_grid_t = "n_grid_t" in d
    if dt is not None:
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError(f"propagators.dt must be positive, got {dt}")
        if has_n_grid_t:
            raise ValueError(
                "propagators: specify exactly one of {'dt', 'n_grid_t'}; "
                "got both."
            )
        n_grid_t = max(2, int(math.ceil(t_max / dt)))
    else:
        n_grid_t = int(d.get("n_grid_t", 60))

    c_closed_form = d.get("c_closed_form", "auto")
    if c_closed_form in (False, None, "none", "null", "off"):
        c_closed_form = None
    elif c_closed_form is True or str(c_closed_form).lower() == "auto":
        c_closed_form = "auto"
    else:
        raise ValueError(
            f"propagators.c_closed_form must be 'auto' or null/false; got "
            f"{c_closed_form!r}.  To supply your own closed form use "
            f"c_closed_form_module / c_closed_form_attr."
        )
    c_method = str(d.get("c_method", "auto"))
    if c_method not in ("auto", "dblquad", "gauss_legendre"):
        raise ValueError(
            f"propagators.c_method must be 'auto', 'dblquad' or "
            f"'gauss_legendre'; got {c_method!r}."
        )

    return PropagatorsConfig(
        t_max=t_max,
        n_grid_t=n_grid_t,
        dt=dt,
        homogeneity=d.get("homogeneity"),
        r_max=d.get("r_max"),
        n_grid_r=d.get("n_grid_r"),
        n_grid_cos=d.get("n_grid_cos"),
        x_max=d.get("x_max"),
        n_grid_x=d.get("n_grid_x"),
        n_jobs=int(d.get("n_jobs", 1)),
        c_closed_form=c_closed_form,
        c_closed_form_module=module_spec,
        c_closed_form_attr=str(d.get("c_closed_form_attr", "C_fn")),
        c_closed_form_only=bool(d.get("c_closed_form_only", False)),
        c_closed_form_vectorized=bool(d.get("c_closed_form_vectorized", False)),
        cache_path=d.get("cache_path"),
        interp_method=str(d.get("interp_method", "linear")),
        c_method=c_method,
        c_n_gauss=int(d.get("c_n_gauss", 20)),
        diag_C=bool(d.get("diag_C", True)),
    )


def _parse_component_axis(d: dict) -> list:
    """``sweep.component_tuples`` (or its older 2-point name
    ``sweep.component_pairs``) as a list of tuples.

    Only the shape of the YAML is checked here; the length of each tuple
    against the observable, and the index range against ``n_components``,
    are checked by :meth:`Expansion.sweep` (and before the expansion runs,
    by :func:`run_workflow` and the ``--dry-run`` estimate).
    """
    given = [k for k in ("component_tuples", "component_pairs")
             if d.get(k) is not None]
    if len(given) == 2:
        raise ValueError(
            "sweep: give component_tuples or component_pairs, not both "
            "(component_pairs is the older name of the same axis)."
        )
    if not given:
        raise ValueError(
            "sweep.component_tuples is required: a list of component-index "
            "tuples, one index per observable operator, e.g. [[0, 1, 1]] "
            "for a 3-point observable (component_pairs, e.g. [[0, 1]], is "
            "the older 2-point spelling)."
        )
    key = given[0]
    raw = d[key]
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            f"sweep.{key} must be a non-empty list of component tuples, "
            f"e.g. [[0, 1]]; got {raw!r}."
        )
    tuples = []
    for entry in raw:
        if not isinstance(entry, (list, tuple)):
            raise ValueError(
                f"sweep.{key}: each entry is a list of component indices, "
                f"one per observable operator, e.g. [0, 1]; got {entry!r}."
            )
        tuples.append(tuple(entry))
    return tuples


def _parse_sweep(d: dict) -> SweepConfig:
    if "positions_grid" not in d:
        raise ValueError("sweep.positions_grid is required")
    if "t_final_grid" not in d:
        raise ValueError("sweep.t_final_grid is required")
    cps = _parse_component_axis(d)
    seed = d.get("seed", 42)
    return SweepConfig(
        positions_grid={k: list(v) for k, v in d["positions_grid"].items()},
        t_final_grid=list(d["t_final_grid"]),
        external_times_grid=(
            {k: list(v) for k, v in d["external_times_grid"].items()}
            if d.get("external_times_grid") else None
        ),
        component_pairs=cps,
        orders=d.get("orders"),
        vertex_types=d.get("vertex_types"),
        integrate_over=d.get("integrate_over"),
        method=str(d.get("method", "qmc_vectorized")),
        n_samples=int(d.get("n_samples", 2 ** 13)),
        seed=None if seed is None else int(seed),
        n_jobs=int(d.get("n_jobs", 1)),
        n_gauss=int(d.get("n_gauss", 8)),
    )


def _parse_output(d: dict) -> OutputConfig:
    if "type" not in d:
        raise ValueError("each output entry must specify a 'type'")
    t = d["type"]
    if t not in ("table", "npz", "plot"):
        raise ValueError(
            f"output type must be one of 'table', 'npz', 'plot'; "
            f"got {t!r}."
        )
    return OutputConfig(
        type=t,
        path=d.get("path"),
        format=str(d.get("format", "markdown")),
        x=d.get("x"),
        y=str(d.get("y", "value")),
        hue=d.get("hue", "order"),
        facet_col=d.get("facet_col"),
    )


# =========================================================================
# WorkflowConfig → L1 System
# =========================================================================


def build_system(cfg: SystemConfig):
    """Lower a parsed :class:`SystemConfig` to a
    :class:`sft_wick.System` instance."""
    from . import specs as sp
    from .system import System

    # Linear operator
    lin_d = dict(cfg.linear)
    lt = lin_d.pop("type", "diagonal")
    if lt == "diagonal":
        gamma = lin_d["gamma"]
        # Pass callables through to DiagonalA; only flatten static lists/arrays.
        gamma_arg = gamma if callable(gamma) else list(gamma)
        diag_kwargs = {"gamma": gamma_arg}
        for k in ("t_max_cache", "n_grid_cache", "t_min_cache"):
            if k in lin_d:
                diag_kwargs[k] = lin_d[k]
        linear = sp.DiagonalA(**diag_kwargs)
    elif lt == "explicit":
        # User-supplied R(t1, t2): bypass the gamma-spline cache entirely.
        # A scalar R (iso_R: true) or an N x N matrix (iso_R: false).
        linear = sp.ExplicitR(
            R_time=lin_d["R_time"],
            iso_R=_as_bool(lin_d.get("iso_R", True), "system.linear.iso_R"),
        )
    else:
        raise ValueError(
            f"Unsupported linear operator type {lt!r}.  "
            f"Supported: 'diagonal', 'explicit'."
        )

    # Noise
    noise = _build_noise(cfg.noise, base_dir=cfg.base_dir,
                         n_components=cfg.n_components)

    vertices = [_build_vertex(sp.LocalVertex, v) for v in cfg.vertices]
    nonlocal_vertices = [_build_vertex(sp.NonLocalVertex, v)
                         for v in cfg.nonlocal_vertices]

    return System(
        field=sp.FieldSpec(cfg.field_name, n_components=cfg.n_components),
        linear=linear,
        noise=noise,
        vertices=tuple(vertices),
        nonlocal_vertices=tuple(nonlocal_vertices),
        t_min=cfg.t_min,
    )


#: Kernel blocks of each separable κ² type, and the axis each is built on.
_SEPARABLE_KAPPA2 = {
    "separable_translation": (("temporal", "time"), ("spatial", "space")),
    "separable_rotation": (("temporal", "time"), ("angular", "angular")),
}


def _build_noise(d: dict, base_dir: Path | None = None,
                 n_components: int | None = None):
    from . import specs as sp

    if not isinstance(d, dict) or not isinstance(d.get("kappa2"), dict):
        raise ValueError(
            "system.noise must be a mapping with a 'kappa2' block and an "
            "optional 'sigma2' block."
        )
    k2_d = dict(d["kappa2"])
    kt = k2_d.pop("type", None)
    if kt in _SEPARABLE_KAPPA2:
        blocks = _SEPARABLE_KAPPA2[kt]
        _reject_unknown_keys(k2_d, [b for b, _ in blocks],
                             f"system.noise.kappa2 (type {kt!r})")
        missing = [b for b, _ in blocks if b not in k2_d]
        if missing:
            raise ValueError(
                f"system.noise.kappa2 of type {kt!r} requires the kernel "
                f"block(s) {missing}."
            )
        kernels = [_build_kernel(k2_d[b], axis=ax, base_dir=base_dir)
                   for b, ax in blocks]
        if kt == "separable_translation":
            kappa2 = sp.SeparableTranslation(temporal=kernels[0],
                                             spatial=kernels[1])
        else:
            kappa2 = sp.SeparableRotation(temporal=kernels[0],
                                          angular=kernels[1])
    elif kt == "callable_module":
        _reject_unknown_keys(k2_d, ("module", "attr"),
                             "system.noise.kappa2 (type 'callable_module')")
        if base_dir is None:
            raise ValueError(
                "noise.kappa2.type='callable_module' requires base_dir; "
                "the workflow loader should pass it through."
            )
        if "module" not in k2_d:
            raise ValueError(
                "noise.kappa2.type='callable_module' requires "
                "'module: <relative path to .py file>'."
            )
        mod_path = (base_dir / k2_d.pop("module")).resolve()
        attr = k2_d.pop("attr", "kappa2")
        fn = _load_callable_from_module(mod_path, attr)
        kappa2 = sp.GeneralKappa2(fn=fn)
    else:
        raise ValueError(
            f"Unsupported kappa2.type {kt!r}.  Supported: "
            f"'separable_translation', 'separable_rotation', 'callable_module'."
        )

    sigma2 = _build_sigma2(d.get("sigma2"), base_dir, n_components)
    return sp.GaussianNoise(kappa2=kappa2, sigma2=sigma2)


def _build_sigma2(sig_d: Any, base_dir: Path | None,
                  n_components: int | None):
    """``system.noise.sigma2``: ``null``, ``{type: constant, amplitude}``
    with a number or an N x N matrix, ``{type: callable_module}``, or
    ``{type: multiplicative, g0, g1, interpretation}``."""
    from . import specs as sp

    if sig_d is None:
        return None
    where = "system.noise.sigma2"
    if not isinstance(sig_d, dict):
        raise ValueError(
            f"{where} must be null or a mapping such as "
            f"{{type: constant, amplitude: 0.01}}; got {sig_d!r}."
        )
    st = sig_d.get("type", "constant")
    if st == "constant":
        _reject_unknown_keys(sig_d, ("type", "amplitude"), where)
        if "amplitude" not in sig_d:
            raise ValueError(
                f"{where}: type 'constant' requires 'amplitude': a number "
                f"(amplitude * I_N) or an N x N matrix as nested lists."
            )
        return sp.ConstantImpulse(amplitude=_sigma2_amplitude(
            sig_d["amplitude"], n_components, f"{where}.amplitude"))
    if st == "callable_module":
        _reject_unknown_keys(sig_d, ("type", "module", "attr"), where)
        if base_dir is None:
            raise ValueError(
                "noise.sigma2.type='callable_module' requires base_dir; "
                "the workflow loader should pass it through."
            )
        if "module" not in sig_d:
            raise ValueError(
                "noise.sigma2.type='callable_module' requires "
                "'module: <relative path to .py file>'."
            )
        mod_path = (base_dir / sig_d["module"]).resolve()
        attr = sig_d.get("attr", "sigma2")
        fn = _load_callable_from_module(mod_path, attr)
        return sp.CustomImpulse(fn=fn)
    if st == "multiplicative":
        # White noise with amplitude g(phi) = g0 + g1 phi; see
        # MultiplicativeImpulse.  D0 = g0 g0^T enters C, the rest lowers to
        # local vertices.
        _reject_unknown_keys(
            sig_d, ("type", "g0", "g1", "interpretation", "vertex_names",
                    "drift_names"), where)
        missing = [k for k in ("g0", "g1") if k not in sig_d]
        if missing:
            raise ValueError(
                f"{where}: type 'multiplicative' requires {missing}: "
                f"g0 as an N x M nested list, g1 as N x M x N."
            )
        names = {k: tuple(sig_d[k]) for k in ("vertex_names", "drift_names")
                 if k in sig_d}
        return sp.MultiplicativeImpulse(
            g0=np.asarray(sig_d["g0"], dtype=float),
            g1=np.asarray(sig_d["g1"], dtype=float),
            interpretation=str(sig_d.get("interpretation", "ito")),
            **names,
        )
    raise ValueError(
        f"Unsupported sigma2.type {st!r}.  Supported: "
        f"'constant', 'callable_module', 'multiplicative'."
    )


def _sigma2_amplitude(raw: Any, n_components: int | None, where: str):
    """A white-noise amplitude: a number, or a symmetric N x N matrix."""
    if isinstance(raw, list):
        try:
            amp = np.asarray(raw, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(
                f"{where} must be a number or an N x N matrix of numbers; "
                f"got {raw!r}."
            ) from None
    else:
        return _yaml_number(raw, where)
    n = n_components
    if (amp.ndim != 2 or amp.shape[0] != amp.shape[1]
            or (n is not None and amp.shape != (n, n))):
        size = f"{n} x {n}" if n is not None else "N x N"
        raise ValueError(
            f"{where} has shape {amp.shape}; a matrix amplitude must be "
            f"{size} (n_components = {n})."
        )
    if not np.all(np.isfinite(amp)):
        raise ValueError(f"{where} has a non-finite entry: {raw!r}.")
    scale = float(np.abs(amp).max())
    if float(np.abs(amp - amp.T).max()) > 1e-12 * scale:
        raise ValueError(
            f"{where} must be symmetric: it is the covariance of the white "
            f"noise, and the action term psi sigma2 psi / 2 reads only its "
            f"symmetric part.  Got {raw!r}."
        )
    return amp


#: Parameters of each built-in kernel of a separable κ², by (axis, type).
_KERNEL_PARAMS = {
    ("time", "exponential"): ("lam", "sigma_t"),
    ("time", "gaussian"): ("lam", "sigma_t"),
    ("space", "exponential"): ("sigma_x",),
    ("space", "gaussian"): ("sigma_x",),
    ("angular", "legendre"): ("coeffs",),
}
#: YAML block of each kernel axis; also the default ``attr`` of a
#: ``type: custom`` kernel.
_KERNEL_BLOCK = {"time": "temporal", "space": "spatial", "angular": "angular"}


def _reject_unknown_keys(d: dict, allowed, where: str) -> None:
    """Refuse a key the block does not read: a misspelt key is otherwise
    ignored without a word, and its default used instead."""
    unknown = sorted(str(k) for k in d if k not in set(allowed))
    if unknown:
        raise ValueError(
            f"{where}: unknown key(s) {unknown}; allowed: "
            f"{sorted(allowed)}."
        )


def _yaml_number(value: Any, where: str):
    """A finite real number from YAML.

    PyYAML reads ``1e-3`` (no decimal point) as a string, so a numeric
    string is converted; any other non-number raises.  A number is
    returned unchanged, so existing specs keep their ``repr``.
    """
    if isinstance(value, bool):
        raise ValueError(f"{where} must be a number; got {value!r}.")
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            raise ValueError(
                f"{where} must be a number; got {value!r}."
            ) from None
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{where} must be a number; got {value!r}.")
    if not math.isfinite(float(value)):
        raise ValueError(f"{where} must be finite; got {value!r}.")
    return value


_TRUE_WORDS = {"true", "yes", "on", "1"}
_FALSE_WORDS = {"false", "no", "off", "0"}


def _as_bool(value: Any, where: str) -> bool:
    """A YAML flag as a bool.

    ``bool("false")`` is ``True``, so a quoted ``"false"`` used to switch a
    flag on; the words true/false, yes/no, on/off and 1/0 are read, and
    anything else raises.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    if isinstance(value, str):
        word = value.strip().lower()
        if word in _TRUE_WORDS:
            return True
        if word in _FALSE_WORDS:
            return False
    raise ValueError(f"{where} must be true or false; got {value!r}.")


def _build_kernel(d: dict, axis: str, base_dir: Path | None = None):
    """One kernel of a separable κ²: a built-in family, or ``type: custom``
    with ``module`` (+ ``attr``) naming a user callable, which becomes a
    :class:`~sft_wick.workflow.specs.CustomKernel`."""
    from . import specs as sp

    where = f"system.noise.kappa2.{_KERNEL_BLOCK[axis]}"
    if not isinstance(d, dict):
        raise ValueError(
            f"{where} must be a mapping such as {{type: exponential, ...}}; "
            f"got {d!r}."
        )
    kt = d.get("type", "exponential")
    if kt == "custom":
        _reject_unknown_keys(d, ("type", "module", "attr"), where)
        if "module" not in d:
            raise ValueError(
                f"{where}: type 'custom' requires 'module: <path to a .py "
                f"file, relative to the YAML file>' exporting the kernel "
                f"callable as 'attr' (default {_KERNEL_BLOCK[axis]!r})."
            )
        if base_dir is None:
            raise ValueError(
                f"{where}: type 'custom' needs the YAML file's directory to "
                f"resolve 'module'; the workflow loader passes it."
            )
        fn = _load_callable_from_module(
            (base_dir / str(d["module"])).resolve(),
            str(d.get("attr", _KERNEL_BLOCK[axis])),
        )
        return sp.CustomKernel(fn=fn)

    params = _KERNEL_PARAMS.get((axis, kt))
    if params is None:
        supported = sorted(t for (a, t) in _KERNEL_PARAMS if a == axis)
        raise ValueError(
            f"Unsupported {axis}-kernel type {kt!r} at {where}.  "
            f"Supported: {supported + ['custom']}."
        )
    _reject_unknown_keys(d, ("type",) + params, where)
    missing = [p for p in params if p not in d]
    if missing:
        raise ValueError(f"{where}: type {kt!r} requires {missing}.")
    if kt == "legendre":
        coeffs = d["coeffs"]
        if not isinstance(coeffs, list) or not coeffs:
            raise ValueError(
                f"{where}.coeffs must be a non-empty list [C_0, C_1, ...]; "
                f"got {coeffs!r}."
            )
        return sp.LegendreAngular(coeffs=[
            _yaml_number(c, f"{where}.coeffs[{i}]")
            for i, c in enumerate(coeffs)
        ])
    vals = {p: _yaml_number(d[p], f"{where}.{p}") for p in params}
    for p in ("sigma_t", "sigma_x"):
        if p in vals and not float(vals[p]) > 0.0:
            raise ValueError(f"{where}.{p} must be positive; got {vals[p]!r}.")
    cls = {
        ("time", "exponential"): sp.ExponentialTemporal,
        ("time", "gaussian"): sp.GaussianTemporal,
        ("space", "exponential"): sp.ExponentialSpatial,
        ("space", "gaussian"): sp.GaussianSpatial,
    }[(axis, kt)]
    return cls(**vals)


# =========================================================================
# Full runner
# =========================================================================


def run_workflow(cfg: WorkflowConfig, progress: Any = None):
    """Execute the full pipeline — expand, build propagators, sweep,
    emit outputs.

    Returns ``(sweep, totals_dataframe)`` for programmatic use.

    ``progress`` is the :mod:`sft_wick.progress` setting for the three
    reported stages (expansion, propagator table, sweep): ``True`` /
    ``False`` / callable / ``None`` (inherit).
    """
    from sft_wick.progress import progress as _progress_scope
    from sft_wick.progress import stage

    with _progress_scope(progress):
        return _run_workflow(cfg, stage)


def _run_workflow(cfg: WorkflowConfig, stage):
    system = build_system(cfg.system)

    # Check the component axis against the observable before the expensive
    # stages run; Expansion.sweep checks it again, but only after the
    # expansion and the propagator table have been built.
    from .expansion import _resolve_component_tuples

    _resolve_component_tuples(
        None, cfg.sweep.component_tuples, tuple(cfg.expand.observable),
        system.n_components, "sweep.component_tuples",
    )

    # ``propagators.diag_C`` is the user-facing knob (the single
    # source of truth for "is C diagonal?"). The symbolic-side
    # ``expand.diag_C`` must agree: with ``propagators.diag_C=False``
    # the closed-form C returns a full (N, N) matrix per sample, but
    # ``expand.diag_C=True`` would collapse the observable (a, b)
    # index pair through ``KroneckerDelta(a, b)`` -- zeroing every
    # cross-component pair at order 0. Reject the contradictory
    # combination with a clear pointer instead of silently rounding
    # the result to zero.
    expand_diag_C = cfg.expand.diag_C
    if (not cfg.propagators.diag_C) and cfg.expand.diag_C:
        # User opted into off-diagonal C but left expand.diag_C at its
        # default (True). Auto-sync so the common case works without
        # forcing users to set the knob twice.
        expand_diag_C = False

    with stage("expansion", f"orders {list(cfg.expand.orders)}"):
        expansion = system.expand(
            observable=cfg.expand.observable,
            orders=cfg.expand.orders,
            response_phase=cfg.expand.response_phase,
            ito=cfg.expand.ito,
            collect_topology=cfg.expand.collect_topology,
            iso_R=cfg.expand.iso_R,
            diag_R=cfg.expand.diag_R,
            diag_C=expand_diag_C,
            iso_C=cfg.expand.iso_C,
            cache_path=cfg.expand.cache_path,
        )
    counts = ", ".join(
        f"order {o}: {len(expansion.diagrams(o))}" for o in expansion.orders
    )
    _log(f"[sft-wick] diagrams -- {counts}")

    c_fn = _load_c_closed_form(cfg.propagators)
    # User-supplied C_fn modules are loaded via
    # :func:`_load_callable_from_module`, which registers them under
    # their ``.py`` file's bare stem and adds the parent directory to
    # ``sys.path`` — so the callable is importable in joblib loky
    # subprocesses. Combined with the module-level
    # ``_ClosedFormPropagatorCache`` class, this lets users opt into
    # parallel C-table builds (``propagators.n_jobs: -1``) when their
    # c_fn does heavy work per call.
    n_jobs = cfg.propagators.n_jobs
    with stage("propagators", f"t_max={cfg.propagators.t_max}, "
                              f"n_grid_t={cfg.propagators.n_grid_t}"):
        props = system.propagators(
            t_max=cfg.propagators.t_max,
            n_grid_t=cfg.propagators.n_grid_t,
            homogeneity=cfg.propagators.homogeneity,
            r_max=cfg.propagators.r_max,
            n_grid_r=cfg.propagators.n_grid_r,
            n_grid_cos=cfg.propagators.n_grid_cos,
            x_max=cfg.propagators.x_max,
            n_grid_x=cfg.propagators.n_grid_x,
            n_jobs=n_jobs,
            c_closed_form=c_fn,
            cache_path=cfg.propagators.cache_path,
            interp_method=cfg.propagators.interp_method,
            c_closed_form_only=cfg.propagators.c_closed_form_only,
            c_closed_form_vectorized=cfg.propagators.c_closed_form_vectorized,
            c_method=cfg.propagators.c_method,
            c_n_gauss=cfg.propagators.c_n_gauss,
            diag_C=cfg.propagators.diag_C,
        )
    _log(f"[sft-wick] C propagator: {props.c_source}"
         + (", lazy per-separation table" if props.is_lazy else ""))

    # Mutual-exclusion: parallelism layers cannot nest because joblib's
    # loky backend does not support nested process pools. Higher-level
    # ``sweep.n_jobs`` (over Cartesian-product grid points) is forwarded
    # to ``expansion.sweep``; lower-level ``expand.n_jobs`` (over
    # diagrams within a single grid point) is forwarded as
    # ``evaluate_n_jobs``. The downstream :meth:`Expansion.sweep` enforces
    # the ``exactly one of {n_jobs, evaluate_n_jobs} > 1`` invariant.
    if int(cfg.expand.n_jobs) != 1 and int(cfg.sweep.n_jobs) != 1:
        raise ValueError(
            "Specify exactly one of {expand.n_jobs > 1, sweep.n_jobs > 1}; "
            "nested joblib loky pools are not supported."
        )

    with stage("sweep", f"method={cfg.sweep.method}"):
        sweep = expansion.sweep(
            props,
            positions_grid=cfg.sweep.positions_grid,
            t_final_grid=cfg.sweep.t_final_grid,
            external_times_grid=cfg.sweep.external_times_grid,
            component_tuples=cfg.sweep.component_tuples,
            orders=cfg.sweep.orders,
            vertex_types=cfg.sweep.vertex_types,
            integrate_over=cfg.sweep.integrate_over,
            method=cfg.sweep.method,
            n_samples=cfg.sweep.n_samples,
            seed=cfg.sweep.seed,
            n_jobs=cfg.sweep.n_jobs,
            evaluate_n_jobs=cfg.expand.n_jobs,
            n_gauss=cfg.sweep.n_gauss,
        )

    totals = sweep.totals()
    for out in cfg.output:
        _emit_output(out, sweep, totals)

    return sweep, totals


def _log(msg: str) -> None:
    """Stage-level notice, printed only when progress reporting is on."""
    from sft_wick.progress import current_setting

    if current_setting() is True:
        print(msg, file=sys.stderr, flush=True)


def _load_c_closed_form(cfg: PropagatorsConfig):
    """The ``c_closed_form`` argument for :meth:`System.propagators`.

    A user module (``c_closed_form_module`` / ``c_closed_form_attr``)
    is imported and returned; otherwise the ``c_closed_form`` setting
    itself (``'auto'`` or ``None``) is passed through.
    """
    if cfg.c_closed_form_module is None:
        return cfg.c_closed_form
    path = Path(cfg.c_closed_form_module).resolve()

    # Register the module under its file basename and put the parent on
    # sys.path AND PYTHONPATH so that joblib loky workers can re-import
    # it when receiving tasks during parallel C-table builds, integration,
    # or sweep dispatch. See ``_load_callable_from_module`` for the same
    # pattern applied to coupling / gamma callables.
    parent_dir = str(path.parent)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    pp = os.environ.get("PYTHONPATH", "")
    pp_parts = pp.split(os.pathsep) if pp else []
    if parent_dir not in pp_parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([parent_dir, *pp_parts])
    module_name = path.stem

    spec_obj = importlib.util.spec_from_file_location(module_name, path)
    if spec_obj is None or spec_obj.loader is None:
        raise ImportError(
            f"Cannot load c_closed_form_module {path!r}."
        )
    module = importlib.util.module_from_spec(spec_obj)
    sys.modules[module_name] = module
    spec_obj.loader.exec_module(module)
    fn = getattr(module, cfg.c_closed_form_attr, None)
    if fn is None:
        raise AttributeError(
            f"Module {path!r} has no attribute "
            f"{cfg.c_closed_form_attr!r}."
        )
    _register_module_by_value(module)
    return fn


def _emit_output(out: OutputConfig, sweep, totals) -> None:
    if out.type == "table":
        payload = _format_table(totals, out.format)
        _write_or_print(payload, out.path)
    elif out.type == "npz":
        if out.path is None:
            raise ValueError("output type 'npz' requires a 'path'.")
        Path(out.path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out.path,
            **{col: totals[col].to_numpy() for col in totals.columns},
        )
    elif out.type == "plot":
        if out.path is None:
            raise ValueError("output type 'plot' requires a 'path'.")
        if out.x is None:
            raise ValueError("output type 'plot' requires 'x'.")
        fig = sweep.plot(
            x=out.x, y=out.y, hue=out.hue, facet_col=out.facet_col,
        )
        Path(out.path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out.path, dpi=120, bbox_inches="tight")


def _format_table(df, fmt: str) -> str:
    if fmt == "markdown":
        return df.to_markdown(index=False)
    if fmt == "csv":
        return df.to_csv(index=False)
    if fmt == "plain":
        return df.to_string(index=False)
    raise ValueError(
        f"output.format must be 'markdown', 'csv', or 'plain'; got {fmt!r}."
    )


def _write_or_print(payload: str, path: Any) -> None:
    if path is None:
        print(payload)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(payload)
