#!/usr/bin/env python3
"""Bounded GL peak-memory experiment with an independent analytic integral.

Run with the sft-wick conda Python::

    python tools/benchmark_gl_memory.py --json gl-memory.json

One Gaussian Wick contribution is C(t0,t1) C(t2,t3) C(t4,t5), with
C(t,s)=t*s and all six times integrated over [0,1]. Its analytic value is
(integral_0^1 t dt)^6 = 1/64. This isolates node/batch allocation, without
graph-generation, interpolator, callable-coupling or convergence claims.
Four fresh subprocesses run sequentially: two modes times separate timing and
tracemalloc phases. Each process covers both node counts; no worker pools run.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import inspect
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
import tempfile
import time
import tracemalloc

REFERENCE = 1.0 / 64.0
DIMENSIONS = 6
SOURCE_FILES = (
    "src/sft_wick/evaluate.py",
    "src/sft_wick/perturbation.py",
    "src/sft_wick/expressions.py",
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
    parser.add_argument("--nodes", type=int, nargs="+", default=[6, 8])
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument(
        "--chunk-kwarg",
        default="gl_chunk_size",
        help="Name of the final GL method's chunk keyword.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=115.0)
    parser.add_argument(
        "--modes", choices=["eager", "chunked"], nargs="+", default=["eager", "chunked"]
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=["eager", "chunked"], help=argparse.SUPPRESS)
    parser.add_argument("--phase", choices=["timing", "memory"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if len(args.nodes) > 2 or any(n not in (6, 8) for n in args.nodes):
        parser.error("--nodes is bounded to at most the two sizes 6 and 8")
    if len(set(args.nodes)) != len(args.nodes):
        parser.error("--nodes must be unique")
    if args.chunk_size < 1 or args.chunk_size > 2**16:
        parser.error("--chunk-size must be in 1..65536")
    if args.repeats < 3 or args.repeats > 5:
        parser.error("--repeats must be in 3..5")
    if not 0 < args.timeout <= 120:
        parser.error("--timeout must be in (0,120]")
    if len(set(args.modes)) != len(args.modes):
        parser.error("--modes must be unique")
    if not args.chunk_kwarg.isidentifier():
        parser.error("--chunk-kwarg must be a Python identifier")
    if args.worker and (args.mode is None or args.phase is None):
        parser.error("worker requires --mode and --phase")
    if not args.worker and args.json is None:
        parser.error("--json is required")
    return args


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rss_bytes():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw * 1024)


class PolynomialCache:
    """Small analytic cache implementing the existing scalar batch contract.

    C(t,s)=t*s is positive semidefinite: phi(t)=t Z, Z~Normal(0,1).
    This benchmark evaluates one Wick pairing, whose weight is one, rather
    than the full six-point Gaussian moment, whose coefficient would be 15.
    """

    def __init__(self, model, np):
        self.model = model
        self.np = np
        self.reset_counts()

    def reset_counts(self):
        self.batch_calls = 0
        self.batch_sizes = set()
        self.samples_across_C_calls = 0

    def C_diagonal_batch(self, t1, t2):
        self.batch_calls += 1
        self.batch_sizes.add(int(len(t1)))
        self.samples_across_C_calls += int(len(t1))
        return (self.np.asarray(t1) * self.np.asarray(t2))[:, None]

    def count_record(self):
        return {
            "C_batch_calls": self.batch_calls,
            "C_batch_sizes": sorted(self.batch_sizes),
            "max_C_batch_size": max(self.batch_sizes, default=0),
            "samples_across_C_calls": self.samples_across_C_calls,
        }


def check_value(estimate, error):
    residual = abs(float(estimate) - REFERENCE)
    return {
        "estimate": float(estimate),
        "reported_error": float(error),
        "analytic_reference": REFERENCE,
        "absolute_residual": residual,
        "passed": math.isfinite(estimate)
        and math.isfinite(error)
        and residual <= 5e-14
        and error == 0.0,
    }


def worker(args):
    os.environ.update(THREAD_ENV)
    sys.path.insert(0, str(args.source.resolve() / "src"))
    rss_before_import = rss_bytes()
    begin = time.perf_counter()
    import numpy as np
    import sft_wick
    from sft_wick.evaluate import PropagatorModel
    from sft_wick.expressions import Propagator, Rational
    from sft_wick.perturbation import DiagramTerm

    import_seconds = time.perf_counter() - begin
    rss_after_import = rss_bytes()
    logs = io.StringIO()
    with redirect_stdout(logs), redirect_stderr(logs):
        begin = time.perf_counter()
        labels = tuple(f"x{i}" for i in range(DIMENSIONS))
        term = DiagramTerm(
            propagators=tuple(
                Propagator("C", None, None, labels[i], labels[i + 1])
                for i in range(0, DIMENSIONS, 2)
            ),
            coupling_sum=Rational(1),
            rational_prefactor=Rational(1),
            integration_vars=(),
            summation_indices=(),
            n_response=0,
        )
        integrand = term.build_integrand({})
        # R and kappa2 are unused: no R factor or generated C table is needed.
        model = PropagatorModel(
            R_time=lambda t, s: 1.0,
            kappa2=lambda n1, t1, n2, t2: np.zeros((1, 1)),
            n_components=1,
            iso_R=True,
            diag_C=True,
        )
        cache = PolynomialCache(model, np)
        setup_seconds = time.perf_counter() - begin
        signature = inspect.signature(integrand.integrate_moment_gauss_legendre)
        if args.mode == "chunked" and args.chunk_kwarg not in signature.parameters:
            raise ValueError(
                f"Chunk keyword {args.chunk_kwarg!r} is not implemented by "
                f"DiagramIntegrand.integrate_moment_gauss_legendre{signature}. "
                "Adapt --chunk-kwarg or finish the production implementation."
            )
        cases = []
        for nodes in sorted(args.nodes):
            kwargs = dict(
                lambda_f=1.0,
                cache=cache,
                t_min=0.0,
                n_gauss=nodes,
                integrate_over="all",
            )
            if args.mode == "chunked":
                kwargs[args.chunk_kwarg] = args.chunk_size
            node_count = nodes**DIMENSIONS
            outputs, samples, durations = [], [], []
            traced = None
            reps = args.repeats if args.phase == "timing" else 1
            for _ in range(reps):
                cache.reset_counts()
                if args.phase == "memory":
                    tracemalloc.start()
                begin = time.perf_counter()
                value, error = integrand.integrate_moment_gauss_legendre(**kwargs)
                durations.append(time.perf_counter() - begin)
                if args.phase == "memory":
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    traced = {
                        "python_traced_current_bytes": current,
                        "python_traced_peak_bytes": peak,
                    }
                outputs.append(check_value(value, error))
                samples.append(cache.count_record())
            expected_batches = (
                1 if args.mode == "eager" else math.ceil(node_count / args.chunk_size)
            )
            counts_pass = all(
                s["C_batch_calls"] == 3 * expected_batches
                and s["samples_across_C_calls"] == 3 * node_count
                and s["max_C_batch_size"]
                <= (node_count if args.mode == "eager" else args.chunk_size)
                for s in samples
            )
            cases.append(
                {
                    "n_gauss": nodes,
                    "dimensions": DIMENSIONS,
                    "tensor_nodes_per_integral": node_count,
                    "mode": args.mode,
                    "phase": args.phase,
                    "chunk_size": args.chunk_size if args.mode == "chunked" else None,
                    "all_duration_seconds": durations,
                    "cold_first_seconds": durations[0],
                    "warm_repeat_seconds": durations[1:],
                    "median_all_seconds": statistics.median(durations),
                    "median_warm_seconds": (
                        statistics.median(durations[1:]) if durations[1:] else None
                    ),
                    "results": outputs,
                    "batch_observations": samples,
                    "batch_counts_passed": counts_pass,
                    "python_tracemalloc": traced,
                    "process_lifetime_peak_rss_bytes": rss_bytes(),
                    "peak_rss_minus_post_import_highwater_bytes": max(
                        0, rss_bytes() - rss_after_import
                    ),
                    "passed": counts_pass and all(o["passed"] for o in outputs),
                }
            )
    return {
        "mode": args.mode,
        "phase": args.phase,
        "source_import": str(Path(sft_wick.__file__).resolve()),
        "numpy_version": np.__version__,
        "import_seconds": import_seconds,
        "fixture_setup_seconds": setup_seconds,
        "rss_before_import_bytes": rss_before_import,
        "rss_after_import_bytes": rss_after_import,
        "GL_signature": str(signature),
        "cases": cases,
        "captured_messages": logs.getvalue()[-1000:],
    }


def coordinator(args):
    source = args.source.resolve()
    if not (source / SOURCE_FILES[0]).is_file():
        raise ValueError(f"No sft-wick source in {source}")
    os.environ.update(THREAD_ENV)
    report = {
        "schema_version": 1,
        "source": str(source),
        "script_sha256": digest(__file__),
        "source_sha256": {p: digest(source / p) for p in SOURCE_FILES},
        "python": sys.version,
        "platform": platform.platform(),
        "thread_environment": THREAD_ENV,
        "settings": {
            "nodes": args.nodes,
            "dimensions": DIMENSIONS,
            "chunk_size": args.chunk_size,
            "chunk_kwarg": args.chunk_kwarg,
            "repeats": args.repeats,
            "modes": args.modes,
        },
        "analytic_reference": {
            "single_diagram": "C01 C23 C45",
            "covariance": "C(t,s)=t*s",
            "integral": "(integral_0^1 t dt)^6 = 1/64",
            "value": REFERENCE,
        },
        "measurement_notes": [
            "Four fresh subprocesses maximum, dispatched sequentially; no joblib pools.",
            "Each process imports once and evaluates both bounded grid sizes.",
            "Serial children share a task-owned temporary Matplotlib cache; imports are timed separately.",
            "The first repeat is cold; the remaining repeats reuse the integrand/cache instance.",
            "The larger size follows the smaller size; its process RSS includes earlier work.",
            "Timing and tracemalloc are separate process phases; traced time is not a speed claim.",
            "ru_maxrss is lifetime process high-water RSS, not an instantaneous allocation count.",
            "RSS minus post-import high-water is coarse and is not a rigorous allocated-byte measure.",
            "tracemalloc omits some native allocations; RSS complements its evidence.",
            "This smooth N=1 scalar diagram isolates tensor-node/batch memory, not physics coverage.",
            "All modes use identical quadrature resolution, diagram, domain, and analytic C.",
            "Tensor-node coverage is audited through C calls; exact node order is a separate unit test.",
            "Reported GL error remains zero; analytic residual is recorded separately.",
        ],
        "workers": [],
        "comparisons": [],
        "complete": False,
    }
    started = time.perf_counter()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        for phase in ("timing", "memory"):
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
                "--phase",
                phase,
                "--nodes",
                *map(str, args.nodes),
                "--chunk-size",
                str(args.chunk_size),
                "--chunk-kwarg",
                args.chunk_kwarg,
                "--repeats",
                str(args.repeats),
                "--timeout",
                str(args.timeout),
            ]
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
                timeout=remaining,
                check=True,
            )
            report["workers"].append(json.loads(proc.stdout))
            report["elapsed_seconds"] = time.perf_counter() - started
            args.json.write_text(json.dumps(report, indent=2) + "\n")
    if set(args.modes) == {"eager", "chunked"}:
        indexed = {
            (w["mode"], w["phase"], c["n_gauss"]): c
            for w in report["workers"]
            for c in w["cases"]
        }
        for nodes in args.nodes:
            eager = indexed["eager", "timing", nodes]
            chunked = indexed["chunked", "timing", nodes]
            et = indexed["eager", "memory", nodes]["python_tracemalloc"]
            ct = indexed["chunked", "memory", nodes]["python_tracemalloc"]
            differences = [
                abs(a["estimate"] - b["estimate"])
                for a, b in zip(eager["results"], chunked["results"])
            ]
            report["comparisons"].append(
                {
                    "n_gauss": nodes,
                    "tensor_nodes_per_integral": nodes**DIMENSIONS,
                    "max_abs_eager_chunk_difference": max(differences),
                    "parity_within_5e_minus14": max(differences) <= 5e-14,
                    "median_seconds_eager_over_chunk": eager["median_all_seconds"]
                    / chunked["median_all_seconds"],
                    "timing_process_peak_rss_eager_bytes": eager[
                        "process_lifetime_peak_rss_bytes"
                    ],
                    "timing_process_peak_rss_chunk_bytes": chunked[
                        "process_lifetime_peak_rss_bytes"
                    ],
                    "python_traced_peak_eager_bytes": et["python_traced_peak_bytes"],
                    "python_traced_peak_chunk_bytes": ct["python_traced_peak_bytes"],
                    "python_traced_peak_eager_over_chunk": et[
                        "python_traced_peak_bytes"
                    ]
                    / max(1, ct["python_traced_peak_bytes"]),
                }
            )
    report["complete"] = True
    report["all_checks_passed"] = all(
        c["passed"] for w in report["workers"] for c in w["cases"]
    ) and all(c["parity_within_5e_minus14"] for c in report["comparisons"])
    report["elapsed_seconds"] = time.perf_counter() - started
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "json": str(args.json.resolve()),
                "complete": True,
                "all_checks_passed": report["all_checks_passed"],
                "elapsed_seconds": report["elapsed_seconds"],
            }
        )
    )
    return 0 if report["all_checks_passed"] else 1


def main():
    args = parse_args()
    if args.worker:
        print(json.dumps(worker(args)))
        return 0
    with tempfile.TemporaryDirectory(prefix="sft-wick-gl-mpl-") as mpl_cache:
        os.environ["MPLCONFIGDIR"] = mpl_cache
        return coordinator(args)


if __name__ == "__main__":
    raise SystemExit(main())
