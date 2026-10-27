#!/usr/bin/env python3
"""Pinned native-word and selected-statistic costs for finite Gaussian families.

Run formal measurements serially in three fresh processes, with six balanced
paired repetitions in each. Preparation of the shared parent family is outside
all query timings. Selected-bank preparation and a genuinely combined
preparation-plus-query interval are reported separately from warm lookup.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import struct
import sys
import time
import tracemalloc

import numpy as np

FROZEN_QUADRATIC_SHA256 = (
    "45fba9523eef34e5ebb1e7f33d6900607163a25f30f9704f00a16115d8653f27"
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_pins(source_root):
    return {
        "src/" + str(path.relative_to(source_root)): _sha(path)
        for path in sorted(source_root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


def _array_pin(value):
    array = np.asarray(value, dtype=np.float64)
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def _bits(values):
    return b"".join(struct.pack("!d", value) for value in values).hex()


def _outcome(call):
    try:
        value = call()
    except (ArithmeticError, ValueError) as error:
        return {"type": type(error).__name__, "args": error.args}
    return {"type": "float", "bits": _bits([value])}


def _load_reference(source_root, expected_sha):
    path = source_root / "sft_wick" / "quadratic.py"
    if _sha(path) != expected_sha:
        raise ValueError("reference quadratic SHA256 differs from the declared pin")
    specification = importlib.util.spec_from_file_location(
        "t003_joint_statistic_reference", path
    )
    if specification is None or specification.loader is None:
        raise ValueError("reference quadratic module could not be loaded")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _data(rank):
    random = np.random.default_rng(732_113 + rank)
    count = 4
    raw = random.normal(size=(count, rank, rank))
    matrices = (raw + raw.swapaxes(-1, -2)) * (0.005 / math.sqrt(max(rank, 1)))
    for index in range(count):
        matrices[index] += np.eye(rank) * (0.05 + 0.01 * index)
    endpoints = random.normal(size=(count, rank)) * (0.05 / math.sqrt(max(rank, 1)))
    constants = np.array([0.05, -0.1, 0.2, 0.0])
    return np.eye(rank), matrices, endpoints, constants


def _family(module, data):
    factor, matrices, endpoints, constants = data
    return module.GaussianQuadraticFamily.from_factor(
        factor, matrices, endpoints, constant=constants
    )


def _prepared_input_pins(parents):
    return {
        str(rank): {
            label: {
                "matrices": _array_pin(family._matrices),
                "endpoints": _array_pin(family._endpoints),
                "constants": _array_pin(family._constants),
            }
            for label, family in (("reference", pair[0]), ("candidate", pair[1]))
        }
        for rank, pair in parents.items()
    }


def _timed(call):
    start = time.perf_counter_ns()
    result = call()
    elapsed = (time.perf_counter_ns() - start) / 1e9
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ArithmeticError("benchmark timing must be finite and positive")
    return elapsed, result


def _peak(call):
    gc.collect()
    tracemalloc.start()
    try:
        result = call()
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {"retained_traced_bytes": current, "peak_traced_bytes": peak}, result


def _paired(reference, candidate, repetitions, expected):
    raw, reference_seconds, candidate_seconds = [], [], []
    for repetition in range(repetitions):
        order = (
            ("reference", "candidate")
            if repetition % 2 == 0
            else ("candidate", "reference")
        )
        timings = {}
        for backend in order:
            elapsed, result = _timed(reference if backend == "reference" else candidate)
            if _bits(result) != expected:
                raise AssertionError("timed query result differs from admitted bits")
            timings[backend] = elapsed
        reference_seconds.append(timings["reference"])
        candidate_seconds.append(timings["candidate"])
        raw.append({"repetition": repetition, "order": order, "seconds": timings})
    reference_median = statistics.median(reference_seconds)
    candidate_median = statistics.median(candidate_seconds)
    return {
        "raw": raw,
        "reference_seconds": reference_seconds,
        "candidate_seconds": candidate_seconds,
        "reference_median_seconds": reference_median,
        "candidate_median_seconds": candidate_median,
        "candidate_change_percent": 100 * (candidate_median / reference_median - 1),
        "reference_over_candidate": reference_median / candidate_median,
    }


def _run_queries(family, queries, repeated_batches):
    result = []
    for _ in range(repeated_batches):
        for method, key in queries:
            result.append(getattr(family, method)(key))
    return result


def _prepare(family, queries):
    return family.prepare_statistics(
        indices=[key for method, key in queries if method == "mixed_cumulant"],
        coefficients=[key for method, key in queries if method == "joint_coefficient"],
    )


def _prepared_then_run(family, queries, repeated_batches):
    return _run_queries(_prepare(family, queries), queries, repeated_batches)


def _query_sets():
    return {
        "one_selected_query": (("mixed_cumulant", (0, 0, 0, 0)),),
        "fourth_order_mixed_and_coefficients": (
            ("mixed_cumulant", (0, 0, 0, 0)),
            ("mixed_cumulant", (0, 0, 1, 1)),
            ("mixed_cumulant", (0, 0, 1, 2)),
            ("mixed_cumulant", (0, 1, 2, 3)),
            ("joint_coefficient", (4, 0, 0, 0)),
            ("joint_coefficient", (2, 2, 0, 0)),
            ("joint_coefficient", (2, 1, 1, 0)),
            ("joint_coefficient", (1, 1, 1, 1)),
        ),
        "low_order_mixed_and_coefficients": (
            ("mixed_cumulant", (0,)),
            ("mixed_cumulant", (0, 1)),
            ("joint_coefficient", (1, 0, 0, 0)),
            ("joint_coefficient", (1, 1, 0, 0)),
        ),
    }


def _deep_key_bytes(value, seen=None):
    """Python object-size estimate, deduplicating shared tuple/int references."""
    if seen is None:
        seen = set()
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, tuple):
        size += sum(_deep_key_bytes(item, seen) for item in value)
    return size


def _bank_storage(bank):
    return {
        "selected_entries": bank.entry_count,
        "float_table_bytes": bank._values.nbytes,
        "key_python_bytes_estimate": _deep_key_bytes(bank._keys),
        "outcome_python_bytes_estimate": _deep_key_bytes(bank._outcomes),
        "scope": (
            "float_table_bytes is only the retained numeric buffer; Python object "
            "estimates include reachable tuples/integers once within each field "
            "and exclude allocator overhead, parent family and model inputs"
        ),
    }


def _exception_admission(reference, candidate):
    checks = []
    cases = [
        ("coefficient_finite_cumulant_overflow", [[1.0]], [[[1e154]]]),
        ("underflow", np.eye(1), [[[1e-120]]]),
    ]
    matrices = np.zeros((2, 64, 64))
    matrices[0, :3, :3] = np.diag([1e150, -1e150, 1.0])
    matrices[1, :3, :3] = np.diag([2e150, -2e150, 3.0])
    cases.append(("exact_work_bound", np.eye(64), matrices))
    for name, factor, matrices in cases:
        old = reference.GaussianQuadraticFamily.from_factor(factor, matrices)
        new = candidate.GaussianQuadraticFamily.from_factor(factor, matrices)
        queries = (
            (("mixed_cumulant", (0, 0, 1)), ("joint_coefficient", (2, 1)))
            if len(matrices) == 2
            else (("mixed_cumulant", (0, 0)), ("joint_coefficient", (2,)))
        )
        bank = _prepare(new, queries)
        for method, key in queries:
            reference_outcome = _outcome(lambda: getattr(old, method)(key))
            expected = reference_outcome
            if name == "exact_work_bound":
                # The frozen reference refused this padded case at its rational
                # work limit; the certified exact contraction returns
                # tr(A*A*B) = 3 exactly, so kappa(0, 0, 1) = 24 and the
                # coefficient of J0**2 J1 is 12.
                exact = 24.0 if method == "mixed_cumulant" else 12.0
                expected = {"type": "float", "bits": _bits([exact])}
            actual = _outcome(lambda: getattr(new, method)(key))
            prepared = _outcome(lambda: getattr(bank, method)(key))
            if not expected == actual == prepared:
                raise AssertionError("numeric exception admission differs")
            checks.append(
                {
                    "case": name,
                    "method": method,
                    "key": key,
                    "outcome": expected,
                    "reference_outcome": reference_outcome,
                }
            )
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-src", required=True, type=Path)
    parser.add_argument("--expect-reference-sha256", required=True)
    parser.add_argument(
        "--candidate-src",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expect-source-manifest", type=Path)
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 2 or args.repetitions % 2:
        raise ValueError("repetitions must be a positive even number of at least two")
    if args.expect_reference_sha256 != FROZEN_QUADRATIC_SHA256:
        raise ValueError("this benchmark requires the final seventh-tranche reference")
    if not args.smoke and args.repetitions != 6:
        raise ValueError("formal mode requires six balanced repetitions")
    if args.output.exists():
        raise FileExistsError("refuse to overwrite a benchmark receipt")
    thread_environment = {
        key: os.environ.get(key)
        for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    }
    if any(value != "1" for value in thread_environment.values()):
        raise ValueError("set all declared numerical thread limits to one")
    reference_root = args.reference_src.resolve()
    candidate_root = args.candidate_src.resolve()
    source_before = {
        "reference": _source_pins(reference_root),
        "candidate": _source_pins(candidate_root),
        "benchmark_sha256": _sha(Path(__file__)),
    }
    manifest_sha = (
        _sha(args.expect_source_manifest) if args.expect_source_manifest else None
    )
    if args.expect_source_manifest:
        expected = json.loads(args.expect_source_manifest.read_text())["files"]
        selected = {
            name: pin["sha256"] if isinstance(pin, dict) else pin
            for name, pin in expected.items()
            if name.startswith("src/")
            or name == "tools/benchmark_joint_statistic_reuse.py"
        }
        actual = {
            **source_before["candidate"],
            "tools/benchmark_joint_statistic_reuse.py": source_before[
                "benchmark_sha256"
            ],
        }
        if selected != actual:
            raise AssertionError("candidate source differs from the requested freeze")
    sys.path.insert(0, str(candidate_root))
    from sft_wick import quadratic as candidate

    if (
        Path(candidate.__file__).resolve()
        != candidate_root / "sft_wick" / "quadratic.py"
    ):
        raise AssertionError("candidate import did not resolve to the declared source")
    reference = _load_reference(reference_root, args.expect_reference_sha256)
    repetitions = 2 if args.smoke else args.repetitions
    ranks = (0, 4) if args.smoke else (0, 4, 32, 128)
    repeated_batches = (1, 2) if args.smoke else (1, 16, 128)
    input_rows, parents = [], {}
    native_rows, selected_rows = [], []
    words = ((0, 0, 0, 0), (0, 0, 1, 1), (0, 0, 1, 2), (0, 1, 2, 3), (0,), (0, 1))
    # All numerical outputs and allowed numeric failures are admitted before
    # any timing. These calls also warm the local numeric kernels explicitly.
    admission = _exception_admission(reference, candidate)
    for rank in ranks:
        data = _data(rank)
        old, new = _family(reference, data), _family(candidate, data)
        parents[rank] = (old, new)
        input_rows.append(
            {
                "rank": rank,
                "factor": _array_pin(data[0]),
                "matrices": _array_pin(data[1]),
                "endpoints": _array_pin(data[2]),
                "constants": _array_pin(data[3]),
            }
        )
        for word in words:
            expected, actual = old.mixed_cumulant(word), new.mixed_cumulant(word)
            if _bits([expected]) != _bits([actual]):
                raise AssertionError("native ordered query bitwise admission differs")
        for queries in _query_sets().values():
            bank = _prepare(new, queries)
            expected = _bits(_run_queries(old, queries, 1))
            if (
                _bits(_run_queries(new, queries, 1)) != expected
                or _bits(_run_queries(bank, queries, 1)) != expected
            ):
                raise AssertionError("selected query bitwise admission differs")
    inputs_sha = hashlib.sha256(
        json.dumps(input_rows, sort_keys=True).encode()
    ).hexdigest()
    prepared_inputs_before = _prepared_input_pins(parents)
    for rank in ranks:
        old, new = parents[rank]
        for word in words:
            expected = _bits([old.mixed_cumulant(word)])

            def old_call():
                return [old.mixed_cumulant(word)]

            def new_call():
                return [new.mixed_cumulant(word)]

            paired = _paired(old_call, new_call, repetitions, expected)
            old_memory, result = _peak(old_call)
            if _bits(result) != expected:
                raise AssertionError("reference memory probe changed output")
            new_memory, result = _peak(new_call)
            if _bits(result) != expected:
                raise AssertionError("candidate memory probe changed output")
            native_rows.append(
                {
                    "rank": rank,
                    "slots": word,
                    "paired": paired,
                    "traced_memory": {"reference": old_memory, "candidate": new_memory},
                    "output_bits": expected,
                }
            )
        for name, queries in _query_sets().items():
            expected_batch = _run_queries(new, queries, 1)
            setup_raw = []
            for repetition in range(repetitions):
                elapsed, bank = _timed(lambda: _prepare(new, queries))
                if _bits(_run_queries(bank, queries, 1)) != _bits(expected_batch):
                    raise AssertionError("standalone preparation changed output")
                setup_raw.append({"repetition": repetition, "seconds": elapsed})
            setup_memory, bank = _peak(lambda: _prepare(new, queries))
            storage = _bank_storage(bank)
            for repeated in repeated_batches:
                expected = _bits(expected_batch * repeated)

                def native():
                    return _run_queries(new, queries, repeated)

                def warm():
                    return _run_queries(bank, queries, repeated)

                def combined():
                    return _prepared_then_run(new, queries, repeated)

                warm_pair = _paired(native, warm, repetitions, expected)
                combined_pair = _paired(native, combined, repetitions, expected)
                memory = {}
                for backend, call in (
                    ("native", native),
                    ("warm_bank", warm),
                    ("prepare_and_lookup", combined),
                ):
                    peak, result = _peak(call)
                    if _bits(result) != expected:
                        raise AssertionError(
                            "selected-query memory probe changed output"
                        )
                    memory[backend] = peak
                selected_rows.append(
                    {
                        "rank": rank,
                        "query_set": name,
                        "queries": queries,
                        "repeated_batches": repeated,
                        "query_count": len(queries) * repeated,
                        "native_reference": "current candidate family, excluding bank preparation",
                        "setup": {
                            "raw": setup_raw,
                            "median_seconds": statistics.median(
                                item["seconds"] for item in setup_raw
                            ),
                            "traced_memory": setup_memory,
                        },
                        "warm_lookup": warm_pair,
                        "prepare_and_lookup_actual_combined": combined_pair,
                        "traced_memory": memory,
                        "retained_storage": storage,
                        "output_bits": expected,
                    }
                )
    source_after = {
        "reference": _source_pins(reference_root),
        "candidate": _source_pins(candidate_root),
        "benchmark_sha256": _sha(Path(__file__)),
    }
    if source_before != source_after:
        raise AssertionError("source or benchmark changed during measurement")
    if args.expect_source_manifest and manifest_sha != _sha(
        args.expect_source_manifest
    ):
        raise AssertionError("expected source manifest changed during measurement")
    prepared_inputs_after = _prepared_input_pins(parents)
    if prepared_inputs_before != prepared_inputs_after:
        raise AssertionError("prepared numeric inputs changed during measurement")
    configuration = io.StringIO()
    with contextlib.redirect_stdout(configuration):
        np.show_config()
    report = {
        "schema": 1,
        "mode": "smoke" if args.smoke else "formal",
        "config": {
            "ranks": ranks,
            "words": words,
            "repeated_batches": repeated_batches,
            "paired_repetitions": repetitions,
            "process_aggregation": "run three fresh processes serially; aggregate their per-process medians",
            "timing_scope": "warmed local kernels after bitwise admission; common immutable parent family preparation excluded; bank setup included only where explicitly labeled; combined interval timed as one actual call",
            "setup_scope": "standalone selected bank construction including validation and both original query kinds; native queries need no additional setup after the shared parent family exists",
            "tracemalloc_scope": "Python and tracked NumPy allocations; excludes untracked native allocations; outputs are retained through peak sampling",
            "rss_scope": "process lifetime maximum, not attributable to a case or backend",
        },
        "source_paths": {
            "reference": str(reference_root),
            "candidate": str(candidate_root),
        },
        "source_before": source_before,
        "source_after": source_after,
        "input_pins": input_rows,
        "input_pins_sha256": inputs_sha,
        "prepared_inputs_before": prepared_inputs_before,
        "prepared_inputs_after": prepared_inputs_after,
        "expected_manifest_sha256": manifest_sha,
        "exception_admission": admission,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": thread_environment,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "numpy_configuration": configuration.getvalue(),
            "max_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "max_rss_units": "bytes" if sys.platform == "darwin" else "KiB",
        },
        "native_rows": native_rows,
        "selected_rows": selected_rows,
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "mode": report["mode"],
                "native_cases": len(native_rows),
                "selected_cases": len(selected_rows),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
