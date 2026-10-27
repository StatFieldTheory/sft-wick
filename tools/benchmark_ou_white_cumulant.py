#!/usr/bin/env python3
"""Source-pinned explicit OU white-leaf versus raw equal-time workflows.

Run formal timings serially. --smoke validates tiny controls only. Both routes
use the same public expansion/integration machinery and bare noise amplitude.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import sft_wick as sw  # noqa: E402
from sft_wick.evaluate import analyze_spatial  # noqa: E402
from sft_wick.white_cumulant import OUWhiteCumulant  # noqa: E402


def hashes():
    paths = sorted((ROOT / "src").rglob("*")) + [Path(__file__).resolve()]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.is_file() and (path.suffix in (".py", ".yaml"))
    }


def array_hash(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def paired(reference, prepared, repetitions, expected=None):
    operations = {"raw": reference, "analytic": prepared}
    timings = {name: [] for name in operations}
    outputs = {name: [] for name in operations} if expected is not None else {}
    orders = []
    for repetition in range(repetitions):
        order = ("raw", "analytic") if repetition % 2 == 0 else ("analytic", "raw")
        orders.append(order)
        for name in order:
            start = time.perf_counter()
            result = operations[name]()
            elapsed = time.perf_counter() - start
            if not np.isfinite(elapsed) or elapsed <= 0:
                raise ArithmeticError("timing must be finite and positive")
            timings[name].append(elapsed)
            if (
                expected is not None
                and float(result).hex() != float(expected[name]).hex()
            ):
                raise AssertionError(
                    "timed workflow result differs from admitted value"
                )
            if expected is not None:
                outputs[name].append(float(result).hex())
    return {
        name: {"seconds": values, "median_seconds": statistics.median(values)}
        for name, values in timings.items()
    } | {"order": orders, "timed_outputs_hex": outputs}


def peak(operation):
    tracemalloc.start()
    try:
        result = operation()
        _, maximum = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, maximum


def base_system(local):
    system = sw.System(
        field=sw.FieldSpec("phi", 2),
        linear=sw.DiagonalA([0.7, 1.3]),
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.2, sigma_t=0.3),
                spatial=sw.ExponentialSpatial(sigma_x=1.0),
            )
        ),
        t_min=0.2,
    )
    if local:
        tensor = np.array([[[0.2, -0.1], [0.3, 0.4]], [[-0.3, 0.2], [0.5, -0.1]]])
        system = replace(system, vertices=(sw.LocalVertex("F", coupling=tensor),))
    return system


def prepare(base, amplitude, observable, order, analytic):
    if analytic:
        leaf = OUWhiteCumulant.from_system(amplitude, base)
        vertex = leaf.as_vertex("K", system=base)
    else:
        vertex = sw.NonLocalVertex("K", order=3, coupling=amplitude, equal_time=True)
    system = replace(base, nonlocal_vertices=(vertex,))
    expansion = system.expand(observable, orders=[order])
    props = system.propagators(
        t_max=1.6,
        n_grid_t=8,
        c_closed_form="auto",
        c_closed_form_only=True,
        progress=False,
    )
    return expansion, props


def evaluate(
    prepared,
    observable,
    components,
    order,
    channel,
    times,
    nodes,
    chunk,
    method="gauss_legendre",
):
    expansion, props = prepared
    positions = {spec.split("(")[1].rstrip(")"): 0.0 for spec in observable}
    return expansion.evaluate(
        props,
        positions=positions,
        t_final=1.1,
        component_pair=components,
        orders=[order],
        vertex_types=[channel],
        method=method,
        n_gauss=nodes if method == "gauss_legendre" else 12,
        n_samples=nodes if method == "qmc_vectorized" else 4096,
        seed=24,
        gl_chunk_size=chunk,
        external_times=times,
    ).total


def run_case(name, nodes, chunks, repetitions, method):
    if name == "K":
        observable = ("phi_a(x)", "phi_b(y)", "phi_c(z)")
        components, order, channel = (0, 1, 1), 1, "K"
        times = {"x": 0.8, "y": 1.1, "z": 0.9}
    elif name == "FK":
        observable = ("phi_a(x)", "phi_b(y)")
        components, order, channel = (0, 1), 2, "FK"
        times = {"x": 0.9, "y": 1.1}
    else:
        observable = tuple(f"phi_{letter}({letter})" for letter in "abcdef")
        components, order, channel = (0, 1, 1, 0, 0, 1), 2, "K"
        times = dict(zip("abcdef", [0.6, 0.9, 1.0, 0.7, 1.1, 0.8]))
    base = base_system(name == "FK")
    amplitude = np.arange(1, 9, dtype=float).reshape((2,) * 3) / 7
    factories = {
        False: lambda: prepare(base, amplitude, observable, order, False),
        True: lambda: prepare(base, amplitude, observable, order, True),
    }
    raw, raw_setup_peak = peak(factories[False])
    analytic, analytic_setup_peak = peak(factories[True])
    setup = paired(factories[False], factories[True], repetitions)
    reference = evaluate(raw, observable, components, order, channel, times, 48, None)
    if not np.isfinite(reference) or abs(reference) < 1e-8:
        raise AssertionError("acceptance channel must be finite and nonzero")

    def call(prepared, count, chunk):
        return evaluate(
            prepared,
            observable,
            components,
            order,
            channel,
            times,
            count,
            chunk,
            method,
        )

    rows = []
    for count in nodes:
        for chunk in chunks:

            def raw_call():
                return call(raw, count, chunk)

            def analytic_call():
                return call(analytic, count, chunk)

            value_raw, peak_raw = peak(raw_call)
            value_analytic, peak_analytic = peak(analytic_call)
            if not np.isfinite(value_raw) or not np.isfinite(value_analytic):
                raise AssertionError("admitted workflow values must be finite")
            tolerance = 2e-9 if method == "gauss_legendre" else 3e-4
            if not np.isclose(value_raw, value_analytic, rtol=tolerance, atol=1e-12):
                raise AssertionError("raw and analytic workflow disagree")
            if not np.isclose(value_analytic, reference, rtol=tolerance, atol=1e-12):
                raise AssertionError("workflow disagrees with raw GL48 refinement")
            expected = {"raw": value_raw, "analytic": value_analytic}
            warm = paired(raw_call, analytic_call, repetitions, expected)
            end_to_end = paired(
                lambda: call(factories[False](), count, chunk),
                lambda: call(factories[True](), count, chunk),
                repetitions,
                expected,
            )
            rows.append(
                {
                    "n_gauss": count if method == "gauss_legendre" else None,
                    "n_samples": count if method == "qmc_vectorized" else None,
                    "method": method,
                    "control_rtol": tolerance,
                    "gl_chunk_size": chunk,
                    "raw_value": value_raw,
                    "analytic_value": value_analytic,
                    "max_abs_error_against_raw": abs(value_raw - value_analytic),
                    "raw_GL48_refinement_abs_error": abs(value_analytic - reference),
                    "raw_route_GL48_refinement_abs_error": abs(value_raw - reference),
                    "warm": warm,
                    "actual_setup_plus_evaluation": end_to_end,
                    "warm_speed_ratio_raw_over_analytic": warm["raw"]["median_seconds"]
                    / warm["analytic"]["median_seconds"],
                    "end_to_end_ratio_raw_over_analytic": end_to_end["raw"][
                        "median_seconds"
                    ]
                    / end_to_end["analytic"]["median_seconds"],
                    "raw_traced_peak_bytes": peak_raw,
                    "analytic_traced_peak_bytes": peak_analytic,
                }
            )
    dimensions = {
        label: sorted(
            {
                len(analyze_spatial(diagram).time_integration_vars)
                for diagram in prepared[0].by_vertex_type(order).get(channel, ())
            }
        )
        for label, prepared in (("raw", raw), ("analytic", analytic))
    }
    return {
        "case": name,
        "components": components,
        "external_times": times,
        "amplitude_sha256": array_hash(amplitude),
        "actual_diagram_dimensions": dimensions,
        "raw_GL48_reference": reference,
        "setup": setup,
        "raw_setup_traced_peak_bytes": raw_setup_peak,
        "analytic_setup_traced_peak_bytes": analytic_setup_peak,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-source-manifest", type=Path)
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument(
        "--method",
        choices=("gauss_legendre", "qmc_vectorized"),
        default="gauss_legendre",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 2 or args.repetitions % 2:
        parser.error("repetitions must be positive even and at least two")
    if not args.smoke:
        if args.repetitions != 6:
            parser.error("formal runs require exactly six balanced repetitions")
        if any(
            os.environ.get(name) != "1"
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        ):
            parser.error("formal runs require explicit single-thread BLAS/OMP settings")
    if args.output.exists():
        raise FileExistsError("refuse to overwrite benchmark receipt")
    before = hashes()
    if args.expect_source_manifest:
        expected = json.loads(args.expect_source_manifest.read_text())["files"]
        selected = {
            name: digest
            for name, digest in expected.items()
            if name.startswith("src/") or name == str(Path(__file__).relative_to(ROOT))
        }
        if selected != before:
            raise RuntimeError("source differs from requested freeze")
    names = ("K", "FK", "K2")
    nodes = (
        ((12,) if args.smoke else (12, 24, 48))
        if args.method == "gauss_legendre"
        else ((8192,) if args.smoke else (8192, 65536))
    )
    chunks = (None,) if args.smoke or args.method == "qmc_vectorized" else (None, 128)
    cases = [
        run_case(name, nodes, chunks, args.repetitions, args.method) for name in names
    ]
    after = hashes()
    if after != before:
        raise RuntimeError("source changed during measurements")
    result = {
        "scope": "declared static equal-time bare tensor and constant diagonal OU response; "
        "finite GL controls only; no continuum, arbitrary callback, PDF or release acceptance",
        "formal": not args.smoke,
        "pid": os.getpid(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "threads": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "PYTHONHASHSEED",
            )
        },
        "source_before": before,
        "source_after": after,
        "source_unchanged": True,
        "cases": cases,
        "max_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "memory_limit": "tracemalloc excludes untracked native allocations; RSS is process-wide; "
        "chunk size limits work blocks, not total process memory",
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "result": "pass",
                "formal": not args.smoke,
                "cases": len(cases),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
