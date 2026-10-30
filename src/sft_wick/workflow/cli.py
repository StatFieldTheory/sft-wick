"""``sft-wick`` CLI: run a full workflow from a YAML config.

Usage::

    sft-wick run CONFIG.yaml
        Execute expansion → propagators → sweep → emit outputs.

    sft-wick run CONFIG.yaml --override sweep.seed=7 --override sweep.n_samples=4096
        Patch config fields without editing the file; a key the file
        leaves out is set too.

    sft-wick run CONFIG.yaml --dry-run
        Parse + validate the config, print a resolved summary and a cost
        estimate (diagram counts, grid size, quadrature calls, a rough
        wall-clock from a one-second micro-benchmark), exit 0.

    sft-wick run CONFIG.yaml --print-resolved
        Also print the config with every default filled in.

    sft-wick run CONFIG.yaml --quiet
        No progress bars or stage banners (``SFT_WICK_PROGRESS=0`` does
        the same from the environment).

    sft-wick validate CONFIG.yaml [--override ...] [--print-resolved]
        Parse the config, build its System and make the checks of the run;
        run nothing.

    sft-wick schema
        Print the JSON Schema of the config files.

    sft-wick quickstart
        Write ``quickstart.yaml`` (a < 1 minute demo1 sweep) to the
        current directory and run it.

A config the parser refuses ends the command with exit status 2 and one
line on stderr, ``sft-wick: config error [code] path: reason``; the codes
are listed under :class:`~sft_wick.workflow.config.ConfigError`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

#: Exit status of a config the parser refuses (argparse uses 2 as well).
CONFIG_ERROR_STATUS = 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sft-wick",
        description=(
            "Run an sft-wick workflow from a YAML config — expand a "
            "perturbative moment, build propagator caches, sweep, "
            "emit tables / arrays / plots."
        ),
    )
    sub = parser.add_subparsers(dest="cmd")

    run_p = sub.add_parser("run", help="Execute a workflow YAML.")
    run_p.add_argument("config", type=Path, help="path to a YAML file")
    run_p.add_argument(
        "--override",
        "-o",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="patch a field, e.g. --override sweep.seed=7.  May be " "repeated.",
    )
    run_p.add_argument(
        "--dry-run",
        action="store_true",
        help="parse and validate the config; print a summary and a cost "
        "estimate; don't execute.",
    )
    run_p.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="no progress bars or stage banners (also: SFT_WICK_PROGRESS=0).",
    )
    run_p.add_argument(
        "--print-resolved",
        action="store_true",
        help="print the config with every default filled in before running.",
    )

    validate_p = sub.add_parser(
        "validate",
        help="parse a workflow YAML and build its System; run nothing.",
    )
    validate_p.add_argument("config", type=Path, help="path to a YAML file")
    validate_p.add_argument(
        "--override",
        "-o",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="patch a field, as for run.  May be repeated.",
    )
    validate_p.add_argument(
        "--print-resolved",
        action="store_true",
        help="print the config with every default filled in, as YAML.",
    )

    sub.add_parser("schema", help="print the JSON Schema of workflow YAML files.")

    compare_p = sub.add_parser(
        "compare-resources",
        help="Compare explicit resource variants in fresh serial processes.",
    )
    compare_p.add_argument("config", type=Path)
    compare_p.add_argument("--variants", type=Path, required=True)
    compare_p.add_argument("--json", dest="report", type=Path, required=True)
    compare_p.add_argument("--repeats", type=int, default=2)
    compare_p.add_argument("--timeout", type=float, default=300.0)
    compare_p.add_argument("--rtol", type=float, default=0.0)
    compare_p.add_argument("--atol", type=float, default=0.0)
    compare_p.add_argument(
        "--force", action="store_true", help="replace an existing comparison report"
    )

    qs_p = sub.add_parser(
        "quickstart",
        help="write quickstart.yaml to the current directory and run it "
        "(< 1 minute on a laptop).",
    )
    qs_p.add_argument(
        "--dir",
        type=Path,
        default=Path("."),
        help="directory to write quickstart.yaml into (default: cwd).",
    )
    qs_p.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing quickstart.yaml.",
    )
    qs_p.add_argument("--quiet", "-q", action="store_true")
    qs_p.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    commands = {
        "run": _cmd_run,
        "validate": _cmd_validate,
        "schema": _cmd_schema,
        "quickstart": _cmd_quickstart,
        "compare-resources": _cmd_compare_resources,
    }
    if args.cmd in commands:
        return commands[args.cmd](args)

    parser.print_help()
    return 1


def _config_error_line(exc: BaseException) -> str:
    """The one line ``sft-wick`` prints for a refused config."""
    from .config import ConfigError

    if isinstance(exc, ConfigError):
        code = exc.code
    elif isinstance(exc, (OSError, ImportError, EOFError)) or _is_yaml_error(exc):
        code = "read_error"
    else:
        code = "invalid_value"
    text = " ".join(str(exc).split())
    return f"sft-wick: config error [{code}] {text}"


def _is_yaml_error(exc: BaseException) -> bool:
    import yaml

    return isinstance(exc, yaml.YAMLError)


def _load(args, *, build: bool = False):
    """Load ``args.config`` with ``args.override``; with ``build``, also
    build the System and make the checks :func:`run_workflow` makes before
    the expansion (:func:`~sft_wick.workflow.config.check_before_run`).

    Returns ``(cfg, overrides, None)``, or ``(None, None, status)`` after
    printing the one-line error.  Each unknown key ignored under
    ``SFT_WICK_STRICT=0`` is printed once as a one-line warning; other
    warnings are passed on.
    """
    import yaml

    from .config import ConfigWarning, check_before_run, load_workflow_config

    caught: list = []
    error = None
    cfg = overrides = None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConfigWarning)
            overrides = _parse_overrides(args.override)
            cfg = load_workflow_config(args.config, overrides=overrides)
            if build:
                check_before_run(cfg)
    except (
        ValueError,
        TypeError,
        ArithmeticError,
        OSError,
        EOFError,
        ImportError,
        yaml.YAMLError,
    ) as exc:
        error = exc
    seen = set()
    for w in caught:
        if not issubclass(w.category, ConfigWarning):
            warnings.showwarning(w.message, w.category, w.filename, w.lineno)
        elif str(w.message) not in seen:
            seen.add(str(w.message))
            print(
                f"sft-wick: config warning [{w.message.code}] {w.message}",
                file=sys.stderr,
            )
    if error is not None:
        print(_config_error_line(error), file=sys.stderr)
        return None, None, CONFIG_ERROR_STATUS
    return cfg, overrides, None


def _resolved_yaml(cfg) -> str:
    import yaml

    from .schema import resolved_config

    return yaml.safe_dump(
        resolved_config(cfg), sort_keys=False, default_flow_style=None, width=88
    )


def _cmd_validate(args) -> int:
    cfg, _, status = _load(args, build=True)
    if status is not None:
        return status
    if args.print_resolved:
        print(_resolved_yaml(cfg), end="")
    else:
        print(f"[sft-wick] valid: {args.config}")
    return 0


def _cmd_schema(args) -> int:
    from .schema import workflow_json_schema

    print(json.dumps(workflow_json_schema(), indent=2))
    return 0


def _progress_setting(args) -> bool:
    """CLI default: report progress (text fallback when piped) unless
    ``--quiet`` or ``SFT_WICK_PROGRESS=0``."""
    if getattr(args, "quiet", False):
        return False
    raw = os.environ.get("SFT_WICK_PROGRESS", "").strip().lower()
    return raw not in ("0", "false", "off", "no")


def _cmd_compare_resources(args) -> int:
    from .resource_compare import compare_resources

    try:
        report = compare_resources(
            args.config,
            args.variants,
            args.report,
            repeats=args.repeats,
            timeout=args.timeout,
            rtol=args.rtol,
            atol=args.atol,
            force=args.force,
        )
    except (ValueError, OSError) as exc:
        print(f"[sft-wick] resource comparison: {exc}", file=sys.stderr)
        return 1
    print("case\tstable\tequivalent\tworkflow seconds\ttraced(load+workflow) MiB")
    for name, case in report["cases"].items():
        seconds, peak = (
            case["median_workflow_seconds"],
            case["median_traced_peak_bytes"],
        )
        print(
            f"{name}\t{case['stable']}\t{case['equivalence_gate_passed']}\t"
            f"{seconds if seconds is not None else '-'}\t"
            f"{peak / 2**20 if peak is not None else '-'}"
        )
    print(f"[sft-wick] report: {args.report}; complete gates passed: {report['ok']}")
    for failure in report["failures"]:
        print(f"[sft-wick] {failure}", file=sys.stderr)
    return 0 if report["ok"] else 1


def _cmd_quickstart(args) -> int:
    from importlib import resources

    target = Path(args.dir) / "quickstart.yaml"
    if target.exists() and not args.force:
        print(
            f"[sft-wick] {target} exists; re-run with --force to overwrite "
            f"or run it directly:  sft-wick run {target}"
        )
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        src = resources.files("sft_wick.workflow").joinpath("quickstart.yaml")
        target.write_text(src.read_text())
        print(f"[sft-wick] wrote {target}")
    args.config = target
    args.override = []
    args.print_resolved = False
    return _cmd_run(args)


def _cmd_run(args) -> int:
    from .config import ConfigError, run_workflow

    t0 = time.perf_counter()
    # Built here once for the checks, so a config error ends with status 2
    # before any stage; run_workflow builds its own System.
    cfg, overrides, status = _load(args, build=True)
    if status is not None:
        return status
    print(
        f"[sft-wick] loaded config: {args.config} " f"({time.perf_counter() - t0:.2f}s)"
    )
    if overrides:
        keys = ", ".join(f"{k}={v!r}" for k, v in overrides.items())
        print(f"[sft-wick] applied overrides: {keys}")
    if getattr(args, "print_resolved", False):
        print("[sft-wick] resolved config:")
        print(_resolved_yaml(cfg), end="")

    _print_summary(cfg)
    _maybe_warn_blas_oversubscription(cfg)
    progress = _progress_setting(args)

    # What only the diagrams decide is refused after the expansion.
    try:
        if args.dry_run:
            from .estimate import estimate_cost

            t0 = time.perf_counter()
            est = estimate_cost(cfg)
            print(est.summary())
            print(
                f"[sft-wick] dry run — exiting before execution "
                f"(estimate took {time.perf_counter() - t0:.1f}s)"
            )
            return 0

        t0 = time.perf_counter()
        sweep, totals = run_workflow(cfg, progress=progress)
    except ConfigError as exc:
        print(_config_error_line(exc), file=sys.stderr)
        return CONFIG_ERROR_STATUS
    print(
        f"[sft-wick] workflow done in "
        f"{time.perf_counter() - t0:.1f}s — "
        f"{len(sweep.rows)} diagram-level rows, "
        f"{len(totals)} aggregated rows."
    )
    return 0


def _parse_overrides(strs: list[str]) -> dict:
    """Parse ``key=value`` strings, auto-coercing values with
    conservative, safe rules (bool → int → float → string).

    No arbitrary-expression parsing: only scalar literals are
    accepted.
    """
    from .config import OverrideError

    out: dict = {}
    for s in strs:
        if "=" not in s:
            raise OverrideError(
                "override_syntax", "", f"--override must be KEY=VALUE, got {s!r}"
            )
        k, v_str = s.split("=", 1)
        out[k.strip()] = _coerce_scalar(v_str.strip())
    return out


def _coerce_scalar(s: str):
    """Try bool → int → float → string."""
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "none", "~"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    # strip surrounding quotes if present
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    return s


def _maybe_warn_blas_oversubscription(cfg) -> None:
    """Print a one-line tip if any layer requested ``n_jobs > 1``
    without one of OPENBLAS / MKL / OMP thread caps set.

    BLAS libraries default to using all cores; combining that with
    ``n_jobs = N_cores`` worker processes yields ``N_cores ** 2``
    threads and is usually slower than running serially. The tip
    suppresses itself once any of the three env vars is set, so users
    who tune their thread budget see no noise.
    """
    requested = (
        int(getattr(cfg.propagators, "n_jobs", 1)) != 1
        or int(getattr(cfg.expand, "n_jobs", 1)) != 1
        or int(getattr(cfg.sweep, "n_jobs", 1)) != 1
    )
    if not requested:
        return
    capped = any(
        os.environ.get(name)
        for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
    )
    if capped:
        return
    print(
        "[sft-wick] tip: n_jobs > 1 was requested. To avoid BLAS "
        "thread oversubscription, set "
        "OPENBLAS_NUM_THREADS=1 / MKL_NUM_THREADS=1 / OMP_NUM_THREADS=1 "
        "before launching. See docs/user_guide/parallelism.rst."
    )


def _print_summary(cfg) -> None:
    s = cfg.system
    print(
        f"[sft-wick] system: field={s.field_name}(N={s.n_components}), "
        f"{len(s.vertices)} local + {len(s.nonlocal_vertices)} non-local "
        f"vertices, linear={s.linear.get('type', 'diagonal')}, "
        f"kappa2={s.noise['kappa2'].get('type', '?')}"
    )
    e = cfg.expand
    print(
        f"[sft-wick] expand: observable={e.observable}, "
        f"orders={list(e.orders)}, target={getattr(e, 'target', 'moment')}"
    )
    p = cfg.propagators
    if p.c_closed_form_module is not None:
        c_src = f"closed_form:user ({p.c_closed_form_module})"
    else:
        c_src = f"c_closed_form={p.c_closed_form!r}, c_method={p.c_method!r}"
    print(
        f"[sft-wick] propagators: t_max={p.t_max}, n_grid_t={p.n_grid_t}, "
        f"n_jobs={p.n_jobs}, {c_src}"
    )
    sw = cfg.sweep
    method_detail = (
        f"n_gauss={sw.n_gauss}"
        if sw.method == "gauss_legendre"
        else f"n_samples={sw.n_samples}, seed={sw.seed}"
    )
    if sw.method == "gauss_legendre":
        gl_bound = "eager" if sw.gl_chunk_size is None else sw.gl_chunk_size
        method_detail += f", gl_chunk_size={gl_bound}"
    if (
        sw.method in {"qmc", "qmc_vectorized"}
        and sw.qmc_coupling_chunk_size is not None
    ):
        method_detail += f", qmc_coupling_chunk_size={sw.qmc_coupling_chunk_size}"
    print(
        f"[sft-wick] sweep: positions_grid={sw.positions_grid}, "
        f"t_final_grid={sw.t_final_grid}, "
        + (
            f"external_times_grid={sw.external_times_grid}, "
            if sw.external_times_grid
            else ""
        )
        + f"component_tuples={sw.component_tuples}, "
        f"integrate_over={sw.integrate_over!r}, "
        f"method={sw.method!r}, {method_detail}"
    )
    for o in cfg.output:
        print(f"[sft-wick] output: type={o.type}, path={o.path}")
    from .resources import resource_controls

    for control in resource_controls(cfg):
        print(
            f"[sft-wick] resource {control['name']}: requested={control['requested']!r}, "
            f"{control['status']} ({control['scope']}); {control['reason']}"
        )


if __name__ == "__main__":
    sys.exit(main())
