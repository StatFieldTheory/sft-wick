#!/usr/bin/env python3
"""Source-pinned prepared-spectrum versus competent dense-Cholesky benchmark.

The default run is a benchmark, not an acceptance test. --smoke uses tiny
controls only. Run formal timings alone, with BLAS/OMP thread counts pinned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import time
import tracemalloc

import numpy as np
import scipy
from scipy.linalg import solve_triangular

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sft_wick.quadratic import GaussianQuadratic  # noqa: E402


def array_hash(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def paired_elapsed(reference, prepared, repetitions):
    """Alternate pair order to avoid giving one implementation the warm slot."""
    operations = {"reference": reference, "prepared": prepared}
    timings = {name: [] for name in operations}
    orders = []
    for repetition in range(repetitions):
        order = ("reference", "prepared")
        if repetition % 2:
            order = order[::-1]
        orders.append(order)
        for name in order:
            start = time.perf_counter()
            operations[name]()
            timings[name].append(time.perf_counter() - start)
    result = {
        name: {"seconds": values, "median_seconds": statistics.median(values)}
        for name, values in timings.items()
    }
    return result["reference"], result["prepared"], orders


def traced_peak(operation):
    tracemalloc.start()
    result = operation()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak


def dense_cgf(sources, matrix, endpoint, constant):
    """One Cholesky per source supplies both log determinant and solve.

    h.T M^-1 h = ||chol(M)^-1 h||^2, so only one triangular solve is needed.
    Covariance factor and B/h are shared inputs, not rebuilt per source.
    """
    results = np.empty(len(sources))
    identity = np.eye(len(endpoint))
    for index, source in enumerate(sources):
        cholesky = np.linalg.cholesky(identity - 2 * source * matrix)
        solved = solve_triangular(cholesky, endpoint, lower=True, check_finite=False)
        results[index] = (
            constant * source
            - np.log(np.diag(cholesky)).sum()
            + 0.5 * source**2 * (solved @ solved)
        )
    return results


def run_case(dimension, counts, chunks, repetitions, seed):
    generator = np.random.default_rng(seed + dimension)
    factor = np.tril(generator.normal(size=(dimension, dimension))) / np.sqrt(dimension)
    factor[np.diag_indices(dimension)] += 2.0
    raw = generator.normal(size=(dimension, dimension))
    quadratic = (raw + raw.T) / (8 * dimension)
    linear = generator.normal(size=dimension) / np.sqrt(dimension)
    constant = 0.2
    matrix = factor.T @ quadratic @ factor
    matrix = matrix / 2 + matrix.T / 2
    endpoint = factor.T @ linear

    def prepare():
        return GaussianQuadratic.from_factor(
            factor, quadratic, linear, constant=constant
        )

    law, setup_peak = traced_peak(prepare)
    reference_setup, setup_timings, setup_order = paired_elapsed(
        lambda: (factor.T @ quadratic @ factor, factor.T @ linear),
        prepare,
        repetitions,
    )
    radius = float(np.max(np.abs(law.eigenvalues), initial=0.0))
    extent = 0.25 / radius if radius else 0.25
    first_source = np.array([extent / 2])
    start = time.perf_counter()
    law.cgf(first_source)
    first_prepared_seconds = time.perf_counter() - start
    start = time.perf_counter()
    dense_cgf(first_source, matrix, endpoint, constant)
    first_dense_seconds = time.perf_counter() - start
    rows = []
    for count in counts:
        sources = np.linspace(-extent, extent, count) if count > 1 else first_source
        reference, dense_peak = traced_peak(
            lambda: dense_cgf(sources, matrix, endpoint, constant)
        )
        for chunk in chunks:
            result, peak = traced_peak(lambda: law.cgf(sources, chunk_size=chunk))
            maximum_error = float(np.max(np.abs(result - reference), initial=0.0))
            if not np.allclose(result, reference, rtol=2e-12, atol=2e-13):
                raise AssertionError(
                    "prepared spectrum disagrees with dense Cholesky reference"
                )
            dense_timings, prepared_timings, timing_order = paired_elapsed(
                lambda: dense_cgf(sources, matrix, endpoint, constant),
                lambda: law.cgf(sources, chunk_size=chunk),
                repetitions,
            )
            rows.append(
                {
                    "source_count": count,
                    "chunk_size": chunk,
                    "sources_sha256": array_hash(sources),
                    "max_abs_cgf_error": maximum_error,
                    "prepared_warm": prepared_timings,
                    "dense_cholesky_warm": dense_timings,
                    "timing_order": timing_order,
                    "prepared_traced_peak_bytes": peak,
                    "dense_traced_peak_bytes": dense_peak,
                    "warm_speed_ratio_dense_over_prepared": dense_timings[
                        "median_seconds"
                    ]
                    / prepared_timings["median_seconds"],
                    "setup_plus_prepared_seconds": setup_timings["median_seconds"]
                    + prepared_timings["median_seconds"],
                    "reference_setup_plus_dense_seconds": reference_setup[
                        "median_seconds"
                    ]
                    + dense_timings["median_seconds"],
                }
            )
    compressed = law.compress(dimension // 4)
    sources = np.linspace(-extent, extent, max(counts))
    error = np.abs(law.cgf(sources) - compressed.cgf(sources))
    bounds = compressed.cgf_error_bound(sources)
    if np.any(error > bounds + 2e-13):
        raise AssertionError(
            "computed spectral-tail error exceeded its conditional bound"
        )
    return {
        "dimension": dimension,
        "input_sha256": {
            "factor": array_hash(factor),
            "quadratic": array_hash(quadratic),
            "linear": array_hash(linear),
            "whitened_B": array_hash(matrix),
            "whitened_h": array_hash(endpoint),
        },
        "constant": constant,
        "reference_real_domain": law.real_domain,
        "setup": setup_timings,
        "reference_B_h_setup": reference_setup,
        "setup_timing_order": setup_order,
        "first_prepared_source_seconds": first_prepared_seconds,
        "first_dense_source_seconds": first_dense_seconds,
        "setup_traced_peak_bytes": setup_peak,
        "prepared_retained_numeric_bytes": law._eigenvalues.nbytes
        + law._linear_squares.nbytes,
        "sweeps": rows,
        "compression_check": {
            "retained_modes": compressed.retained_dimension,
            "retained_numeric_bytes": compressed._eigenvalues.nbytes
            + compressed._linear_squares.nbytes,
            "max_abs_cgf_error": float(np.max(error, initial=0.0)),
            "max_conditional_cgf_error_bound": float(np.max(bounds, initial=0.0)),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny correctness controls; no performance acceptance",
    )
    parser.add_argument("--dimensions", type=int, nargs="+", default=[8, 128])
    parser.add_argument(
        "--source-counts", type=int, nargs="+", default=[1, 16, 256, 1000]
    )
    parser.add_argument(
        "--chunk-sizes", type=int, nargs="+", default=[1, 32, 256, 1000]
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument(
        "--expect-module-sha256",
        help="refuse timings after an unexpected module change",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.smoke:
        args.dimensions, args.source_counts, args.chunk_sizes, args.repetitions = (
            [6],
            [1, 4],
            [1, 4],
            1,
        )
    if (
        min(
            args.dimensions + args.source_counts + args.chunk_sizes + [args.repetitions]
        )
        < 1
    ):
        parser.error(
            "dimensions, source counts, chunk sizes and repetitions must be positive"
        )
    paths = [ROOT / "src/sft_wick/quadratic.py", Path(__file__).resolve()]
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    if (
        args.expect_module_sha256
        and args.expect_module_sha256 != hashes["src/sft_wick/quadratic.py"]
    ):
        parser.error("quadratic module hash differs from the requested source pin")
    report = {
        "mode": (
            "tiny correctness smoke" if args.smoke else "serial formal timing candidate"
        ),
        "source_sha256": hashes,
        "configuration": {
            "dimensions": args.dimensions,
            "source_counts": args.source_counts,
            "chunk_sizes": args.chunk_sizes,
            "repetitions": args.repetitions,
            "seed": args.seed,
        },
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "threads": {
                name: os.environ.get(name)
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
        },
        "comparison": "same finite B/h law; one Cholesky factor and one triangular solve per real source",
        "memory_limit": "tracemalloc excludes untracked native BLAS allocations; process max RSS is global, not per case",
        "scope": "selected finite quadratic Gaussian law; no nonlinear-model, continuum, PDF, package-wide or release acceptance",
        "cases": [
            run_case(
                n, args.source_counts, args.chunk_sizes, args.repetitions, args.seed
            )
            for n in args.dimensions
        ],
    }
    raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["process_max_rss_bytes"] = (
        raw_rss if platform.system() == "Darwin" else raw_rss * 1024
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "mode": report["mode"],
                "source_sha256": hashes,
                "maximum_dense_reference_error": max(
                    row["max_abs_cgf_error"]
                    for case in report["cases"]
                    for row in case["sweeps"]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
