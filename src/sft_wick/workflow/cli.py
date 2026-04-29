"""``sft-wick`` CLI — run a full workflow from a YAML config.

Usage::

    sft-wick run CONFIG.yaml
        Execute expansion → propagators → sweep → emit outputs.

    sft-wick run CONFIG.yaml --override sweep.seed=7 --override sweep.n_samples=4096
        Patch config fields without editing the file.

    sft-wick run CONFIG.yaml --dry-run
        Parse + validate the config, print a resolved summary, exit 0.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


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
        "--override", "-o", action="append", default=[],
        metavar="KEY=VALUE",
        help="patch a field, e.g. --override sweep.seed=7.  May be "
             "repeated.",
    )
    run_p.add_argument(
        "--dry-run", action="store_true",
        help="parse and validate the config; print a summary; "
             "don't execute.",
    )

    args = parser.parse_args(argv)
    if args.cmd == "run":
        return _cmd_run(args)

    parser.print_help()
    return 1


def _cmd_run(args) -> int:
    from .config import load_workflow_config, run_workflow

    overrides = _parse_overrides(args.override)

    t0 = time.perf_counter()
    cfg = load_workflow_config(args.config, overrides=overrides)
    print(f"[sft-wick] loaded config: {args.config} "
          f"({time.perf_counter() - t0:.2f}s)")
    if overrides:
        keys = ", ".join(f"{k}={v!r}" for k, v in overrides.items())
        print(f"[sft-wick] applied overrides: {keys}")

    _print_summary(cfg)
    _maybe_warn_blas_oversubscription(cfg)

    if args.dry_run:
        print("[sft-wick] dry run — exiting before execution")
        return 0

    t0 = time.perf_counter()
    sweep, totals = run_workflow(cfg)
    print(f"[sft-wick] workflow done in "
          f"{time.perf_counter() - t0:.1f}s — "
          f"{len(sweep.rows)} diagram-level rows, "
          f"{len(totals)} aggregated rows.")
    return 0


def _parse_overrides(strs: list[str]) -> dict:
    """Parse ``key=value`` strings, auto-coercing values with
    conservative, safe rules (bool → int → float → string).

    Deliberately avoids any arbitrary-expression parsing — only
    scalar literals are accepted.
    """
    out: dict = {}
    for s in strs:
        if "=" not in s:
            raise SystemExit(
                f"--override must be KEY=VALUE, got {s!r}."
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
    who deliberately tune their thread budget see no noise.
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
    print(f"[sft-wick] system: field={s.field_name}(N={s.n_components}), "
          f"{len(s.vertices)} local + {len(s.nonlocal_vertices)} non-local "
          f"vertices, linear={s.linear.get('type', 'diagonal')}, "
          f"kappa2={s.noise['kappa2'].get('type', '?')}")
    e = cfg.expand
    print(f"[sft-wick] expand: observable={e.observable}, "
          f"orders={list(e.orders)}")
    p = cfg.propagators
    print(f"[sft-wick] propagators: t_max={p.t_max}, n_grid_t={p.n_grid_t}, "
          f"n_jobs={p.n_jobs}, "
          f"c_closed_form={p.c_closed_form_module is not None}")
    sw = cfg.sweep
    method_detail = (
        f"n_gauss={sw.n_gauss}"
        if sw.method == "gauss_legendre"
        else f"n_samples={sw.n_samples}, seed={sw.seed}"
    )
    print(f"[sft-wick] sweep: positions_grid={sw.positions_grid}, "
          f"t_final_grid={sw.t_final_grid}, "
          f"component_pairs={sw.component_pairs}, "
          f"integrate_over={sw.integrate_over!r}, "
          f"method={sw.method!r}, {method_detail}")
    for o in cfg.output:
        print(f"[sft-wick] output: type={o.type}, path={o.path}")


if __name__ == "__main__":
    sys.exit(main())
