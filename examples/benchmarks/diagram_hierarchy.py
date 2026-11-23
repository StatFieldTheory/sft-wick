"""Time two independent solvers for connected cumulants of the same SDE."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
from itertools import product
from math import fsum, isfinite, lgamma, log, sqrt
from pathlib import Path
from statistics import median
from time import perf_counter

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
KERNELS = ("white", "ou", "matern32", "gaussian")
DIMENSIONS = (2, 4, 8, 12)
ORDERS = (1, 2)
CUMULANTS = (2, 3, 4)
BACKENDS = ("diagram", "hierarchy")
THREADS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
REL_TOL = 1e-8
REFINEMENT_TOL = REL_TOL / 10
ROUNDING = 64 * sys.float_info.epsilon


def make_case(kernel, dimension, k, n):
    """One physical site with a quadratic hub coupled to every component."""
    drift = [[0, 0, 0, 0.18]]
    for a in range(1, dimension):
        strength = (0.07 + 0.04 * a / (dimension - 1)) / sqrt(dimension)
        drift.extend(
            [
                [0, 0, a, strength],
                [0, a, 0, 0.6 * strength],
                [a, 0, 0, 0.05 * (1 + 0.2 * a / dimension) / sqrt(dimension)],
            ]
        )
    return {
        "kernel": kernel,
        "dimension": dimension,
        "k": k,
        "n": n,
        "gamma": [0.8 + 0.5 * a / (dimension - 1) for a in range(dimension)],
        "drift": drift,
        "amplitude": 0.05,
        "sigma": 2.3,
        "times": [1.3] * n,
        "components": [0] * n,
        "t_min": 0.15,
    }


def prepare_diagram(case, nodes):
    """Return the native diagram calculation for a fixed quadrature rule."""
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np

    import sft_wick as sw

    if not Path(sw.__file__).resolve().is_relative_to(ROOT / "src"):
        raise RuntimeError("diagram worker imported a different checkout")
    # Match the hierarchy worker's untimed solver imports without
    # constructing a model or running a numerical evaluation here.
    import_module("sft_wick.workflow.closed_forms")
    start = perf_counter()
    families = {
        "white": sw.ExponentialTemporal,
        "ou": sw.ExponentialTemporal,
        "matern32": sw.Matern32Temporal,
        "gaussian": sw.GaussianTemporal,
    }
    dimension = case["dimension"]
    strength = case["amplitude"]
    white = case["kernel"] == "white"
    temporal = families[case["kernel"]](
        lam=0.0 if white else strength, sigma_t=case["sigma"]
    )
    noise = sw.GaussianNoise(
        sw.SeparableTranslation(temporal=temporal, spatial=sw.ExponentialSpatial(1.0)),
        sigma2=sw.ConstantImpulse(strength * np.eye(dimension)) if white else None,
    )
    drift = np.zeros((dimension,) * 3)
    for target, left, right, coefficient in case["drift"]:
        drift[target, left, right] = coefficient
    system = sw.System(
        field=sw.FieldSpec("phi", dimension),
        linear=sw.DiagonalA(case["gamma"]),
        vertices=(sw.LocalVertex("F", coupling=drift),),
        noise=noise,
        t_min=case["t_min"],
    )
    points = [f"x{i}" for i in range(case["n"])]
    horizon = max(case["times"])
    expansion = system.expand(
        tuple(f"phi_{chr(97 + i)}({name})" for i, name in enumerate(points)),
        orders=[case["k"]],
        target="connected",
        progress=False,
    )
    cache = system.propagators(t_max=horizon, c_closed_form_only=True, progress=False)
    if cache.c_source != "closed_form:builtin":
        raise RuntimeError(f"expected built-in covariance, got {cache.c_source}")

    def evaluate():
        result = expansion.evaluate(
            cache,
            positions=dict.fromkeys(points, 0.0),
            t_final=horizon,
            external_times={
                name: time
                for name, time in zip(points, case["times"])
                if time != horizon
            },
            component_pair=tuple(case["components"]),
            method="gauss_legendre",
            n_gauss=nodes,
            n_jobs=1,
        )
        total = float(result.total)
        scale = fsum(abs(row["value"]) for row in result.per_diagram)
        return total, scale

    return evaluate, start


def worker(payload):
    """Evaluate a single backend and return separate phase timings."""
    sys.path.insert(0, str(ROOT))
    case = payload["case"]
    backend = payload["backend"]
    if backend == "hierarchy":
        from _crossover_reference import prepare

        start = perf_counter()
        ref = prepare(case, payload["resolution"])
        evaluate = ref.evaluate
        hidden_states = ref.hidden_states
    else:
        evaluate, start = prepare_diagram(case, payload["resolution"])
        hidden_states = None
    prepared = perf_counter()
    value, scale = evaluate()
    end = perf_counter()
    if backend == "hierarchy" and "sft_wick" in sys.modules:
        raise RuntimeError("hierarchy worker imported the diagram package")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        rss *= 1024
    return {
        "backend": backend,
        "resolution": payload["resolution"],
        "hidden_states": hidden_states,
        "value": float(value),
        "scale": float(scale),
        "setup_seconds": prepared - start,
        "evaluate_seconds": end - prepared,
        "compute_seconds": end - start,
        "peak_rss_bytes": rss,
        "pid": os.getpid(),
    }


def worker_environment(temporary):
    """Run every worker without inherited user configuration or thread counts."""
    return {
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "TMPDIR": temporary,
        "PYTHONPYCACHEPREFIX": str(Path(temporary) / "bytecode"),
        "PYTHONHASHSEED": "0",
        "SFT_WICK_QUIET_CACHE": "1",
        **dict.fromkeys(THREADS, "1"),
    }


def fresh_run(case, backend, resolution, timeout):
    request = json.dumps({"case": case, "backend": backend, "resolution": resolution})
    with tempfile.TemporaryDirectory(prefix="cumulant-benchmark-") as temporary:
        start = perf_counter()
        try:
            result = subprocess.run(
                [sys.executable, "-B", str(SCRIPT), "_worker"],
                cwd=ROOT,
                input=request,
                text=True,
                capture_output=True,
                timeout=timeout,
                env=worker_environment(temporary),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "backend": backend,
                "resolution": resolution,
                "status": "timeout",
                "wall_seconds": perf_counter() - start,
            }
        wall_seconds = perf_counter() - start
    if result.returncode:
        return {
            "backend": backend,
            "resolution": resolution,
            "status": "error",
            "returncode": result.returncode,
            "error": result.stderr[-2000:],
        }
    try:
        out = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {
            "backend": backend,
            "resolution": resolution,
            "status": "error",
            "error": "worker returned an invalid record",
        }
    return dict(out, status="ok", wall_seconds=wall_seconds)


def compare(left, right, tolerance):
    error = abs(left["value"] - right["value"])
    floor = ROUNDING * max(left["scale"], right["scale"])
    return {
        "absolute_error": error,
        "relative_error": error / abs(right["value"])
        if abs(right["value"]) > floor
        else None,
        "threshold": tolerance * abs(right["value"]) + floor,
        "pass": error <= tolerance * abs(right["value"]) + floor,
    }


def gaussian_modes(case):
    """Choose an even spectral rule using the uniform finite-horizon bound."""
    ratio = (max(case["times"]) - case["t_min"]) / case["sigma"]
    for nodes in range(2, 34, 2):
        log_bound = lgamma(nodes + 1) - lgamma(2 * nodes + 1) + 2 * nodes * log(ratio)
        if log_bound <= log(REFINEMENT_TOL):
            return nodes
    raise ValueError("Gaussian kernel did not meet the refinement target with 32 modes")


def calibrate(case, timeout):
    """Select a converged resolution before comparing performance."""
    if case["kernel"] == "gaussian":
        modes = gaussian_modes(case)
        resolutions = {"hierarchy": (modes, modes + 2, modes + 4)}
    else:
        resolutions = {"hierarchy": (None,)}
    resolutions["diagram"] = (20, 28, 40, 56)
    valid, history = {}, {}
    for backend in BACKENDS:
        prior = None
        history[backend] = []
        for resolution in resolutions[backend]:
            run = fresh_run(case, backend, resolution, timeout)
            history[backend].append(run)
            if run["status"] != "ok":
                break
            if len(resolutions[backend]) == 1 or (
                prior is not None and compare(prior, run, REFINEMENT_TOL)["pass"]
            ):
                valid[backend] = (prior or run, run)
                break
            prior = run
    return valid, history


def run_cell(case, repeats, timeout):
    selected, calibration = calibrate(case, timeout)
    row = {"case": case, "calibration": calibration, "runs": [], "timed_pairs": []}
    if len(selected) != len(BACKENDS):
        errors = any(
            run["status"] == "error"
            for history in calibration.values()
            for run in history
        )
        return dict(row, status="worker_error" if errors else "unavailable")
    reference = selected["hierarchy"][1]
    accuracy = compare(selected["diagram"][1], reference, REL_TOL)
    row["comparison"] = accuracy
    if not accuracy["pass"]:
        return dict(row, status="mismatch")
    resolutions = {
        backend: items[0]["resolution"] for backend, items in selected.items()
    }
    row["resolution"] = resolutions
    for repeat in range(repeats):
        pair = {}
        for backend in BACKENDS if repeat % 2 == 0 else reversed(BACKENDS):
            run = fresh_run(case, backend, resolutions[backend], timeout)
            row["runs"].append(dict(run, repeat=repeat))
            if run["status"] != "ok":
                return dict(
                    row,
                    status="unavailable"
                    if run["status"] == "timeout"
                    else "worker_error",
                )
            comparison = compare(run, reference, REL_TOL)
            row["runs"][-1]["comparison"] = comparison
            if not comparison["pass"]:
                return dict(row, status="mismatch")
            pair[backend] = run
        comparison = compare(pair["diagram"], pair["hierarchy"], REL_TOL)
        row["timed_pairs"].append(dict(comparison, repeat=repeat))
        if not comparison["pass"]:
            return dict(row, status="mismatch")
    row["median_compute_seconds"] = {
        backend: median(
            run["compute_seconds"] for run in row["runs"] if run["backend"] == backend
        )
        for backend in BACKENDS
    }
    row["state_dimension"] = case["dimension"] * (
        1 + selected["hierarchy"][0]["hidden_states"]
    )
    row["nonzero"] = abs(reference["value"]) > ROUNDING * reference["scale"]
    return dict(row, status="pass")


def fingerprint():
    """Identify both solvers, the benchmark code and reference inputs."""
    digest = sha256()
    for directory in (
        "src/sft_wick",
        "validation",
        "examples/reference",
        "examples/benchmarks",
    ):
        for file in sorted((ROOT / directory).rglob("*.py")):
            digest.update(file.relative_to(ROOT).as_posix().encode())
            digest.update(file.read_bytes())
    return digest.hexdigest()


def write_checkpoint(payload, output):
    """Replace the checkpoint using a file created exclusively for this write."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=output.parent,
        prefix=f".{output.name}.",
        encoding="utf-8",
        delete=False,
    ) as handle:
        path = Path(handle.name)
        try:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.close()
            path.replace(output)
        finally:
            path.unlink(missing_ok=True)


def record_grid(cases, output, repeats, timeout):
    stamp = fingerprint()
    payload = {
        "fingerprint": stamp,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "versions": {name: version(name) for name in ("numpy", "scipy", "pandas")},
        "requested_threads": dict.fromkeys(THREADS, "1"),
        "planned_cells": [
            [case[key] for key in ("kernel", "dimension", "k", "n")] for case in cases
        ],
        "complete": False,
        "relative_tolerance": REL_TOL,
        "refinement_tolerance": REFINEMENT_TOL,
        "repeats": repeats,
        "timeout_seconds": timeout,
        "rows": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        row = run_cell(case, repeats, timeout)
        payload["rows"].append(row)
        write_checkpoint(payload, output)
        print(
            f"{case['kernel']:8s} D={case['dimension']:2d} "
            f"k={case['k']} n={case['n']} {row['status']}",
            flush=True,
        )
    if fingerprint() != stamp:
        raise RuntimeError("benchmark source changed during measurement")
    statuses = [row["status"] for row in payload["rows"]]
    usable = sum(row["status"] == "pass" and row["nonzero"] for row in payload["rows"])
    payload["complete"] = True
    payload["comparable_nonzero_cells"] = usable
    write_checkpoint(payload, output)
    expected_nonzero = any((case["k"] + case["n"]) % 2 == 0 for case in cases)
    return int(
        "pass" not in statuses
        or (expected_nonzero and not usable)
        or any(s in statuses for s in ("mismatch", "worker_error"))
    )


def main():
    if sys.argv[1:] == ["_worker"]:
        print(json.dumps(worker(json.load(sys.stdin)), allow_nan=False))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full", action="store_true", help="default to the full factorial grid"
    )
    parser.add_argument("--kernels", nargs="+", choices=KERNELS, default=KERNELS)
    parser.add_argument("--dimensions", nargs="+", type=int, choices=DIMENSIONS)
    parser.add_argument("--orders", nargs="+", type=int, choices=ORDERS)
    parser.add_argument("--cumulants", nargs="+", type=int, choices=CUMULANTS)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="seconds allowed per worker"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 2 or not isfinite(args.timeout) or args.timeout <= 0:
        parser.error("use at least two repeats and a positive timeout")
    dims = args.dimensions or (DIMENSIONS if args.full else (2,))
    orders = args.orders or (ORDERS if args.full else (2,))
    cumulants = args.cumulants or (CUMULANTS if args.full else (2,))
    cases = [
        make_case(*values) for values in product(args.kernels, dims, orders, cumulants)
    ]
    return record_grid(cases, args.output, args.repeats, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
