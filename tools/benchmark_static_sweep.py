#!/usr/bin/env python3
"""Bounded public-API benchmark for sweep-local static integrand reuse.

Example (run with the sft-wick conda Python)::

    python tools/benchmark_static_sweep.py --json static-sweep.json

Each mode/method/grid case runs in a fresh, serial subprocess. Main wall-time
measurements exclude profiling/tracemalloc; additional passes measure preparation
and Python allocation peaks. These are performance/parity observations, not an
independent scientific accuracy or convergence test.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc
from functools import wraps

STAGES = (
    "_prepare_sweep",
    "build_coupling_values",
    "build_integrand",
    "analyze_spatial",
    "evaluate_coupling",
    "integrate_moment",
)
SOURCE_FILES = (
    "src/sft_wick/evaluate.py",
    "src/sft_wick/perturbation.py",
    "src/sft_wick/workflow/expansion.py",
    "src/sft_wick/workflow/specs.py",
)
THREAD_ENV = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "SFT_WICK_PROGRESS": "0",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "LOKY_MAX_CPU_COUNT": "4",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--json", type=Path)
    parser.add_argument("--counts", nargs="+", type=int, default=[1, 16, 64])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["gauss_legendre", "qmc_vectorized"],
        default=["gauss_legendre", "qmc_vectorized"],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-gauss", type=int, default=6)
    parser.add_argument("--samples", type=int, default=2**10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--timeout",
        type=float,
        default=115.0,
        help="Overall subprocess time budget in seconds.",
    )
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=["fresh", "reuse"], help=argparse.SUPPRESS)
    parser.add_argument("--count", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--method", choices=["gauss_legendre", "qmc_vectorized"], help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("--repeats must be at least 3 (cold plus two warm repeats)")
    if any(c not in (1, 16, 64) for c in args.counts):
        parser.error("--counts are bounded to 1, 16, 64 geometry configurations")
    if args.n_gauss not in (6, 8):
        parser.error("--n-gauss is bounded to 6 or 8")
    if args.samples != 2**10:
        parser.error("--samples is bounded to 1024 for this benchmark")
    if args.timeout <= 0 or args.timeout > 120:
        parser.error("--timeout must be positive and at most 120 seconds")
    if not args.worker and args.json is None:
        parser.error("--json is required")
    if args.worker and (
        args.mode is None or args.count not in (1, 16, 64) or args.method is None
    ):
        parser.error("worker requires --mode, --count and --method")
    return args


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rss_bytes():
    """ru_maxrss is lifetime high-water RSS, bytes on macOS and KiB on Linux."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw * 1024)


class StageProfiler:
    """Nested measured functions: exclusive subtracts all measured children."""

    def __init__(self):
        self.stack = []
        self.records = {
            s: {"calls": 0, "inclusive_seconds": 0.0, "exclusive_seconds": 0.0}
            for s in STAGES
        }

    def wrap(self, name, original):
        @wraps(original)
        def measured(*args, **kwargs):
            frame = [time.perf_counter(), 0.0]
            self.stack.append(frame)
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - frame[0]
                self.stack.pop()
                record = self.records[name]
                record["calls"] += 1
                record["inclusive_seconds"] += elapsed
                record["exclusive_seconds"] += max(0.0, elapsed - frame[1])
                if self.stack:
                    self.stack[-1][1] += elapsed

        return measured

    @contextmanager
    def installed(self, sw):
        from sft_wick import evaluate, perturbation

        targets = [
            (sw.Expansion, "_prepare_sweep"),
            (sw.System, "build_coupling_values"),
            (perturbation.DiagramTerm, "build_integrand"),
            (evaluate, "analyze_spatial"),
            (perturbation.DiagramTerm, "evaluate_coupling"),
            (evaluate, "integrate_moment"),
        ]
        originals = [(owner, name, getattr(owner, name)) for owner, name in targets]
        try:
            for owner, name, original in originals:
                setattr(owner, name, self.wrap(name, original))
            yield self
        finally:
            for owner, name, original in originals:
                setattr(owner, name, original)


def result_payload(result):
    keys = (
        "x",
        "y",
        "t_final",
        "a",
        "b",
        "order",
        "diagram_idx",
        "vertex_type",
        "n_cross_C",
    )
    return {
        "keys": [[row[k] for k in keys] for row in result.rows],
        "values": [float(row["value"]) for row in result.rows],
        "errors": [float(row["error"]) for row in result.rows],
    }


def parity(left, right):
    keys_equal = left["keys"] == right["keys"]
    if not keys_equal or len(left["values"]) != len(right["values"]):
        return {"same_rows": False, "passed": False}
    dv = [abs(a - b) for a, b in zip(left["values"], right["values"])]
    de = [abs(a - b) for a, b in zip(left["errors"], right["errors"])]
    finite = all(
        math.isfinite(v)
        for v in left["values"] + right["values"] + left["errors"] + right["errors"]
    )
    return {
        "same_rows": True,
        "row_count": len(dv),
        "all_finite": finite,
        "max_abs_value_difference": max(dv, default=0.0),
        "max_abs_error_difference": max(de, default=0.0),
        "passed": finite and not any(dv) and not any(de),
    }


def worker(args):
    os.environ.update(THREAD_ENV)
    sys.path.insert(0, str(args.source.resolve() / "src"))
    started = time.perf_counter()
    import numpy as np
    import scipy
    import sft_wick as sw

    import_seconds = time.perf_counter() - started
    logs = io.StringIO()
    with redirect_stdout(logs), redirect_stderr(logs):
        begin = time.perf_counter()
        # No index symmetry: tests must not accidentally rely on diagonal F.
        F = np.array([[[0.25, -0.40], [0.15, 0.30]], [[-0.20, 0.10], [0.35, -0.45]]])
        system = sw.System(
            field=sw.FieldSpec("phi", n_components=2),
            linear=sw.DiagonalA(gamma=[1.0, 1.0]),
            vertices=[sw.LocalVertex("F", coupling=F)],
            noise=sw.GaussianNoise(
                kappa2=sw.SeparableTranslation(
                    temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
                    spatial=sw.ExponentialSpatial(sigma_x=1.0),
                )
            ),
        )
        system_seconds = time.perf_counter() - begin
        begin = time.perf_counter()
        expansion = system.expand(
            ("phi_a(x)", "phi_b(y)"), orders=[0, 2], progress=False
        )
        expansion_seconds = time.perf_counter() - begin
        begin = time.perf_counter()
        # Exact built-in closed form removes lazy-spline first-use effects.
        props = system.propagators(
            t_max=2.0,
            c_closed_form="auto",
            c_closed_form_only=True,
            c_closed_form_vectorized=True,
            progress=False,
        )
        propagator_seconds = time.perf_counter() - begin
        side = math.isqrt(args.count)
        kwargs = dict(
            positions_grid={"x": [0.0], "y": np.linspace(0.1, 1.1, side).tolist()},
            t_final_grid=np.linspace(0.4, 1.8, side).tolist(),
            component_pairs=[(0, 0), (0, 1), (1, 1)],
            orders=[0, 2],
            method=args.method,
            n_gauss=args.n_gauss,
            n_samples=args.samples,
            seed=args.seed,
            n_jobs=1,
            evaluate_n_jobs=1,
            progress=False,
            reuse_integrands=(args.mode == "reuse"),
        )
        times, baseline, repeat_parity = [], None, []
        for _ in range(args.repeats):
            begin = time.perf_counter()
            result = expansion.sweep(props, **kwargs)
            times.append(time.perf_counter() - begin)
            payload = result_payload(result)
            if baseline is None:
                baseline = payload
            else:
                repeat_parity.append(parity(baseline, payload))
            del result, payload
        # Fresh bindings on every pass; this is warm Python/cache execution,
        # not a warm persistent integrand cache (the new cache is sweep-local).
        profiler = StageProfiler()
        begin = time.perf_counter()
        with profiler.installed(sw):
            profiled = expansion.sweep(props, **kwargs)
        profile_seconds = time.perf_counter() - begin
        profile_parity = parity(baseline, result_payload(profiled))
        del profiled
        memory = None
        if not args.no_memory:
            tracemalloc.start()
            begin = time.perf_counter()
            memory_result = expansion.sweep(props, **kwargs)
            memory_seconds = time.perf_counter() - begin
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory = {
                "seconds_with_tracemalloc": memory_seconds,
                "python_traced_current_bytes": current,
                "python_traced_peak_bytes": peak,
                "parity": parity(baseline, result_payload(memory_result)),
            }
            del memory_result
    return {
        "mode": args.mode,
        "method": args.method,
        "geometry_configurations": args.count,
        "component_pairs": 3,
        "evaluation_points": args.count * 3,
        "diagram_counts": {str(k): len(expansion.diagrams(k)) for k in (0, 2)},
        "source_import": str(Path(sw.__file__).resolve()),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "setup": {
            "import_seconds": import_seconds,
            "system_seconds": system_seconds,
            "expansion_seconds": expansion_seconds,
            "propagators_seconds": propagator_seconds,
            "c_source": props.c_source,
        },
        "sweep_seconds": times,
        "cold_first_sweep_seconds": times[0],
        "median_all_sweeps_seconds": statistics.median(times),
        "warm_repeat_seconds": times[1:],
        "median_warm_repeats_seconds": statistics.median(times[1:]),
        "repeat_parity": repeat_parity,
        "stage_profile": {
            "separate_sweep_seconds": profile_seconds,
            "stages": profiler.records,
            "parity": profile_parity,
        },
        "memory": memory,
        "process_lifetime_peak_rss_bytes": rss_bytes(),
        "captured_messages": logs.getvalue()[-2000:],
        "payload": baseline,
    }


def coordinator(args):
    os.environ.update(THREAD_ENV)
    source = args.source.resolve()
    if not (source / "src/sft_wick/workflow/expansion.py").is_file():
        raise ValueError(f"No sft-wick source in {source}")
    started = time.perf_counter()
    report = {
        "schema_version": 1,
        "source": str(source),
        "source_sha256": {p: digest(source / p) for p in SOURCE_FILES},
        "script_sha256": digest(__file__),
        "python": sys.version,
        "platform": platform.platform(),
        "thread_environment": THREAD_ENV,
        "settings": {
            "counts": args.counts,
            "methods": args.methods,
            "repeats": args.repeats,
            "n_gauss": args.n_gauss,
            "n_samples": args.samples,
            "seed": args.seed,
            "orders": [0, 2],
            "n_components": 2,
        },
        "measurement_notes": [
            "Each mode/method/grid case has a fresh serial subprocess.",
            "Main sweep medians include all repeats; first and warm times are separate.",
            "Warm means reused Python/propagator state; bindings rebuild each sweep.",
            "Stage profile is an additional sweep, excluded from main timing medians.",
            "Inclusive stages overlap; exclusive stages subtract measured nested children.",
            "tracemalloc is a separate additional sweep and misses some native allocations.",
            "ru_maxrss is lifetime peak including imports, setup and all passes; not sweep-only.",
            "Vectorized closed-form-only C avoids interpolation and lazy-table warmup differences.",
            "Parity is against fresh integration at the same settings; no independent accuracy claim.",
            "No model order or integrator resolution was changed between compared modes.",
        ],
        "cases": [],
        "complete": False,
    }
    for method in args.methods:
        for count in args.counts:
            modes = []
            for mode in ("fresh", "reuse"):
                remaining = args.timeout - (time.perf_counter() - started)
                if remaining <= 0:
                    raise TimeoutError("Overall benchmark time budget exhausted")
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--source",
                    str(source),
                    "--mode",
                    mode,
                    "--count",
                    str(count),
                    "--method",
                    method,
                    "--repeats",
                    str(args.repeats),
                    "--n-gauss",
                    str(args.n_gauss),
                    "--samples",
                    str(args.samples),
                    "--seed",
                    str(args.seed),
                    "--timeout",
                    str(args.timeout),
                ]
                if args.no_memory:
                    command.append("--no-memory")
                proc = subprocess.run(
                    command,
                    text=True,
                    capture_output=True,
                    env=os.environ.copy(),
                    timeout=remaining,
                    check=True,
                )
                modes.append(json.loads(proc.stdout))
            difference = parity(modes[0].pop("payload"), modes[1].pop("payload"))
            old = modes[0]["median_all_sweeps_seconds"]
            new = modes[1]["median_all_sweeps_seconds"]
            report["cases"].append(
                {
                    "method": method,
                    "geometry_configurations": count,
                    "modes": modes,
                    "per_diagram_parity": difference,
                    "measured_median_ratio_fresh_over_reuse": old / new,
                }
            )
            # Save completed cases after each pair; a later timeout retains evidence.
            report["elapsed_seconds"] = time.perf_counter() - started
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report, indent=2) + "\n")
    report["complete"] = True
    report["all_parity_checks_passed"] = all(
        case["per_diagram_parity"]["passed"]
        and all(
            all(p["passed"] for p in mode["repeat_parity"])
            and mode["stage_profile"]["parity"]["passed"]
            and (mode["memory"] is None or mode["memory"]["parity"]["passed"])
            for mode in case["modes"]
        )
        for case in report["cases"]
    )
    report["elapsed_seconds"] = time.perf_counter() - started
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "json": str(args.json.resolve()),
                "complete": True,
                "all_parity_checks_passed": report["all_parity_checks_passed"],
                "elapsed_seconds": report["elapsed_seconds"],
            }
        )
    )
    return 0 if report["all_parity_checks_passed"] else 1


def main():
    args = parse_args()
    if args.worker:
        print(json.dumps(worker(args)))
        return 0
    return coordinator(args)


if __name__ == "__main__":
    raise SystemExit(main())
