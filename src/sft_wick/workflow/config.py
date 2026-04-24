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
import sys
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class PropagatorsConfig:
    t_max: float
    n_grid_t: int = 60
    homogeneity: Any = None
    r_max: Any = None
    n_grid_r: Any = None
    n_grid_cos: Any = None
    x_max: Any = None
    n_grid_x: Any = None
    n_jobs: int = 1
    c_closed_form_module: Any = None
    c_closed_form_attr: str = "C_fn"
    cache_path: Any = None


@dataclass(frozen=True)
class SweepConfig:
    positions_grid: dict
    t_final_grid: list
    component_pairs: list
    orders: Any = None
    vertex_types: Any = None
    integrate_over: Any = None
    method: str = "qmc_vectorized"
    n_samples: int = 2 ** 13
    seed: int = 42


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


def _apply_override(data: dict, dotted_key: str, value: Any) -> None:
    """Apply a ``"a.b.c": value`` override to a nested dict."""
    parts = dotted_key.split(".")
    cur = data
    for p in parts[:-1]:
        if not isinstance(cur, dict) or p not in cur:
            raise KeyError(
                f"override key '{dotted_key}' does not exist in config"
            )
        cur = cur[p]
    leaf = parts[-1]
    if not isinstance(cur, dict) or leaf not in cur:
        raise KeyError(
            f"override key '{dotted_key}' does not exist in config"
        )
    cur[leaf] = value


def _parse_workflow(data: dict, base_dir: Path) -> WorkflowConfig:
    system_d = _require_dict(data, "system")
    expand_d = _require_dict(data, "expand")
    props_d = _require_dict(data, "propagators")
    sweep_d = _require_dict(data, "sweep")
    output_d = data.get("output", [])

    system_cfg = _parse_system(system_d, base_dir)
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


def _parse_system(d: dict, base_dir: Path) -> SystemConfig:
    fld = d.get("field", {}) or {}
    name = fld.get("name", "phi")
    nc = int(fld.get("n_components", 1))

    linear = d.get("linear")
    if linear is None:
        raise ValueError("system.linear is required")

    noise = d.get("noise")
    if noise is None:
        raise ValueError("system.noise is required")

    vertices = d.get("vertices", []) or []
    nonlocal_vertices = d.get("nonlocal_vertices", []) or []

    # Resolve coupling tensor file paths relative to the YAML file.
    vertices = [_resolve_coupling(v, base_dir) for v in vertices]
    nonlocal_vertices = [
        _resolve_coupling(v, base_dir) for v in nonlocal_vertices
    ]

    return SystemConfig(
        field_name=name, n_components=nc,
        linear=linear, noise=noise,
        vertices=vertices, nonlocal_vertices=nonlocal_vertices,
        t_min=float(d.get("t_min", 0.0)),
    )


def _resolve_coupling(v: dict, base_dir: Path) -> dict:
    """Resolve the vertex spec's ``coupling`` to either an inline
    numpy array or a callable loaded from a user module.

    Priority order:
      ``coupling``          — inline tensor (nested YAML lists).
      ``coupling_path``     — path to an ``.npy`` file, loaded as a
                              numpy array.
      ``coupling_module``   — path to a ``.py`` module exporting an
                              attribute (default ``coupling_fn``)
                              used as a callable ``fn(n_list,
                              t_list) -> tensor``.  Required for
                              spacetime-dependent non-local vertices
                              like demo2's ``κ^{(3)}``.
    """
    out = dict(v)
    if "coupling_path" in out and "coupling" not in out:
        p = (base_dir / out.pop("coupling_path")).resolve()
        out["coupling"] = np.load(p)
    elif "coupling_module" in out and "coupling" not in out:
        mod_path = (base_dir / out.pop("coupling_module")).resolve()
        attr = out.pop("coupling_attr", "coupling_fn")
        out["coupling"] = _load_callable_from_module(mod_path, attr)
    return out


def _load_callable_from_module(path: Path, attr: str):
    """Import ``path`` as a standalone module and return
    ``getattr(module, attr)``."""
    spec_obj = importlib.util.spec_from_file_location(
        f"_sft_wick_coupling_{path.stem}", path,
    )
    if spec_obj is None or spec_obj.loader is None:
        raise ImportError(
            f"Cannot load coupling module {path!r}."
        )
    module = importlib.util.module_from_spec(spec_obj)
    sys.modules[spec_obj.name] = module
    spec_obj.loader.exec_module(module)
    fn = getattr(module, attr, None)
    if fn is None or not callable(fn):
        raise AttributeError(
            f"Module {path!r} has no callable attribute {attr!r}."
        )
    return fn


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
    )


def _parse_propagators(d: dict, base_dir: Path) -> PropagatorsConfig:
    if "t_max" not in d:
        raise ValueError("propagators.t_max is required")
    module_spec = d.get("c_closed_form_module")
    if module_spec is not None:
        module_spec = str((base_dir / module_spec).resolve())
    return PropagatorsConfig(
        t_max=float(d["t_max"]),
        n_grid_t=int(d.get("n_grid_t", 60)),
        homogeneity=d.get("homogeneity"),
        r_max=d.get("r_max"),
        n_grid_r=d.get("n_grid_r"),
        n_grid_cos=d.get("n_grid_cos"),
        x_max=d.get("x_max"),
        n_grid_x=d.get("n_grid_x"),
        n_jobs=int(d.get("n_jobs", 1)),
        c_closed_form_module=module_spec,
        c_closed_form_attr=str(d.get("c_closed_form_attr", "C_fn")),
        cache_path=d.get("cache_path"),
    )


def _parse_sweep(d: dict) -> SweepConfig:
    if "positions_grid" not in d:
        raise ValueError("sweep.positions_grid is required")
    if "t_final_grid" not in d:
        raise ValueError("sweep.t_final_grid is required")
    if "component_pairs" not in d:
        raise ValueError("sweep.component_pairs is required")
    cps = [tuple(pair) for pair in d["component_pairs"]]
    return SweepConfig(
        positions_grid={k: list(v) for k, v in d["positions_grid"].items()},
        t_final_grid=list(d["t_final_grid"]),
        component_pairs=cps,
        orders=d.get("orders"),
        vertex_types=d.get("vertex_types"),
        integrate_over=d.get("integrate_over"),
        method=str(d.get("method", "qmc_vectorized")),
        n_samples=int(d.get("n_samples", 2 ** 13)),
        seed=int(d.get("seed", 42)),
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
        linear = sp.DiagonalA(gamma=list(lin_d["gamma"]))
    else:
        raise ValueError(
            f"Unsupported linear operator type {lt!r}.  "
            f"Supported: 'diagonal'."
        )

    # Noise
    noise = _build_noise(cfg.noise)

    def _coupling_value(v: dict):
        c = v["coupling"]
        # If already a callable (from coupling_module), pass through.
        return c if callable(c) else np.asarray(c)

    vertices = [
        sp.LocalVertex(name=v["name"], coupling=_coupling_value(v))
        for v in cfg.vertices
    ]
    nonlocal_vertices = [
        sp.NonLocalVertex(
            name=v["name"], order=int(v["order"]),
            coupling=_coupling_value(v),
        )
        for v in cfg.nonlocal_vertices
    ]

    return System(
        field=sp.FieldSpec(cfg.field_name, n_components=cfg.n_components),
        linear=linear,
        noise=noise,
        vertices=tuple(vertices),
        nonlocal_vertices=tuple(nonlocal_vertices),
        t_min=cfg.t_min,
    )


def _build_noise(d: dict):
    from . import specs as sp

    k2_d = dict(d["kappa2"])
    kt = k2_d.pop("type")
    if kt == "separable_translation":
        temporal = _build_kernel(k2_d["temporal"], axis="time")
        spatial = _build_kernel(k2_d["spatial"], axis="space")
        kappa2 = sp.SeparableTranslation(temporal=temporal, spatial=spatial)
    elif kt == "separable_rotation":
        temporal = _build_kernel(k2_d["temporal"], axis="time")
        angular = _build_kernel(k2_d["angular"], axis="angular")
        kappa2 = sp.SeparableRotation(temporal=temporal, angular=angular)
    else:
        raise ValueError(
            f"Unsupported kappa2.type {kt!r}.  Supported: "
            f"'separable_translation', 'separable_rotation'."
        )

    sigma2 = None
    sig_d = d.get("sigma2")
    if sig_d is not None:
        st = dict(sig_d).pop("type", "constant")
        if st == "constant":
            sigma2 = sp.ConstantImpulse(
                amplitude=sig_d.get("amplitude", 0.0)
            )
        else:
            raise ValueError(
                f"Unsupported sigma2.type {st!r}.  Supported: 'constant'."
            )

    return sp.GaussianNoise(kappa2=kappa2, sigma2=sigma2)


def _build_kernel(d: dict, axis: str):
    from . import specs as sp

    kt = d.get("type", "exponential")
    if axis == "time":
        if kt == "exponential":
            return sp.ExponentialTemporal(lam=d["lam"], sigma_t=d["sigma_t"])
        if kt == "gaussian":
            return sp.GaussianTemporal(lam=d["lam"], sigma_t=d["sigma_t"])
    elif axis == "space":
        if kt == "exponential":
            return sp.ExponentialSpatial(sigma_x=d["sigma_x"])
        if kt == "gaussian":
            return sp.GaussianSpatial(sigma_x=d["sigma_x"])
    elif axis == "angular":
        if kt == "legendre":
            return sp.LegendreAngular(coeffs=list(d["coeffs"]))
    raise ValueError(
        f"Unsupported {axis}-kernel type {kt!r}."
    )


# =========================================================================
# Full runner
# =========================================================================


def run_workflow(cfg: WorkflowConfig):
    """Execute the full pipeline — expand, build propagators, sweep,
    emit outputs.

    Returns ``(sweep, totals_dataframe)`` for programmatic use.
    """
    system = build_system(cfg.system)

    expansion = system.expand(
        observable=cfg.expand.observable,
        orders=cfg.expand.orders,
        response_phase=cfg.expand.response_phase,
        ito=cfg.expand.ito,
        collect_topology=cfg.expand.collect_topology,
        iso_R=cfg.expand.iso_R,
        diag_R=cfg.expand.diag_R,
        diag_C=cfg.expand.diag_C,
        iso_C=cfg.expand.iso_C,
        cache_path=cfg.expand.cache_path,
    )

    c_fn = _load_c_closed_form(cfg.propagators)
    # User-supplied C_fn modules are imported via ``importlib.util``
    # and their closures are not picklable across joblib's ``loky``
    # workers.  The closed form is also so fast (microseconds per
    # call vs dblquad's ~100 ms) that parallelism is pointless.
    # Force serial in that case.
    n_jobs = cfg.propagators.n_jobs if c_fn is None else 1
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
    )

    sweep = expansion.sweep(
        props,
        positions_grid=cfg.sweep.positions_grid,
        t_final_grid=cfg.sweep.t_final_grid,
        component_pairs=cfg.sweep.component_pairs,
        orders=cfg.sweep.orders,
        vertex_types=cfg.sweep.vertex_types,
        integrate_over=cfg.sweep.integrate_over,
        method=cfg.sweep.method,
        n_samples=cfg.sweep.n_samples,
        seed=cfg.sweep.seed,
    )

    totals = sweep.totals()
    for out in cfg.output:
        _emit_output(out, sweep, totals)

    return sweep, totals


def _load_c_closed_form(cfg: PropagatorsConfig):
    """Import a user-supplied ``C_fn(n1, t1, n2, t2)`` from the
    ``.py`` file given by ``c_closed_form_module`` in the config.

    Returns ``None`` if the field isn't set.
    """
    if cfg.c_closed_form_module is None:
        return None
    path = Path(cfg.c_closed_form_module)
    spec_obj = importlib.util.spec_from_file_location(
        f"_sft_wick_c_{path.stem}", path,
    )
    if spec_obj is None or spec_obj.loader is None:
        raise ImportError(
            f"Cannot load c_closed_form_module {path!r}."
        )
    module = importlib.util.module_from_spec(spec_obj)
    sys.modules[spec_obj.name] = module
    spec_obj.loader.exec_module(module)
    fn = getattr(module, cfg.c_closed_form_attr, None)
    if fn is None:
        raise AttributeError(
            f"Module {path!r} has no attribute "
            f"{cfg.c_closed_form_attr!r}."
        )
    return fn


def _emit_output(out: OutputConfig, sweep, totals) -> None:
    if out.type == "table":
        payload = _format_table(totals, out.format)
        _write_or_print(payload, out.path)
    elif out.type == "npz":
        if out.path is None:
            raise ValueError("output type 'npz' requires a 'path'.")
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
