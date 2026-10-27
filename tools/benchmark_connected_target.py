"""Bounded graph-generation costs; no numerical integration is performed."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
from statistics import median
from time import perf_counter
import sys

import numpy as np
import sft_wick.perturbation as candidate

from sft_wick import Action, Field, Vertex, compute_moment, compute_moment_numerical
from sft_wick.indices import IndexContext
from sft_wick.vertices import VertexInstance


def _array_receipt(before, after):
    before, after = np.asarray(before), np.asarray(after)
    assert before.dtype == after.dtype and before.shape == after.shape
    assert before.tobytes() == after.tobytes()
    return repr(
        (before.dtype.str, before.shape, hashlib.sha256(before.tobytes()).hexdigest())
    )


def _load_baseline(path):
    spec = importlib.util.spec_from_file_location(
        "sft_wick._connected_benchmark_baseline", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _symbols(expr):
    from sft_wick.expressions import Product, Sum, Symbol

    if isinstance(expr, Symbol):
        return [expr]
    if isinstance(expr, Product):
        return [symbol for factor in expr.factors for symbol in _symbols(factor)]
    if isinstance(expr, Sum):
        return [symbol for term in expr.terms for symbol in _symbols(term)]
    return []


def _reference_owners(obs, vertex, order):
    """Prepare immutable ownership once per order, outside timed filtering."""
    q = len(obs)
    ctx = IndexContext()
    instances = [VertexInstance.instantiate(vertex, ctx, k) for k in range(order)]
    owners = {op.spatial_arg: k for k, op in enumerate(obs)}
    assert len(owners) == q
    for k, instance in enumerate(instances, q):
        for op in instance.field_operators:
            assert owners.get(op.spatial_arg, k) == k
            owners[op.spatial_arg] = k
    return owners


def _late_connected(dt, obs, vertex, order, base_owners):
    """Independent DFS using owning vertex instances, not spatial adjacency.

    Bounded cases use one vertex type. Local instances retain one site each;
    numerical nonlocal instances retain their original leg labels. Symbolic
    nonlocal coupling terms may relabel whole leg groups, so every term's
    symbol arguments supply its vertex incidence separately. A mixed record
    is refused rather than approximated by a representative.
    """
    from sft_wick.expressions import Sum

    q = len(obs)

    def connected(owners):
        graph = [set() for _ in range(q + order)]
        for prop in dt.propagators:
            a, b = owners[prop.spatial_left], owners[prop.spatial_right]
            graph[a].add(b)
            graph[b].add(a)
        seen, pending = {0}, [0]
        while pending:
            for other in graph[pending.pop()]:
                if other not in seen:
                    seen.add(other)
                    pending.append(other)
        return len(seen) == len(graph)

    if vertex.local or order == 0:
        return connected(base_owners)
    terms = (
        dt.coupling_sum.terms
        if isinstance(dt.coupling_sum, Sum)
        else (dt.coupling_sum,)
    )
    verdicts = []
    for term in terms:
        symbols = _symbols(term)
        if not symbols or any(not symbol.spatial_args for symbol in symbols):
            # Numeric generation uses static symbols but the original
            # VertexInstance leg allocation remains attached to the R edges.
            verdicts.append(connected(base_owners))
            continue
        assert len(symbols) == order
        owners = {op.spatial_arg: k for k, op in enumerate(obs)}
        for k, symbol in enumerate(symbols, q):
            assert len(symbol.spatial_args) == vertex.n_fields
            for label in symbol.spatial_args:
                assert label not in owners
                owners[label] = k
        verdicts.append(connected(owners))
    assert (
        len(set(verdicts)) == 1
    ), "bounded comparator encountered a mixed coupling record"
    return verdicts[0]


def run(repeats, baseline_source):
    baseline_source = baseline_source.resolve()
    baseline = _load_baseline(baseline_source)
    phi, psi = Field("phi", "physical", 2), Field("psi", "response", 2)
    F = Vertex([psi, phi, phi], "F")
    H = Vertex([psi, psi, phi, phi], "H")
    K3 = Vertex([psi] * 3, "K3", local=False)
    K4 = Vertex([psi] * 4, "K4", local=False)
    cases = [
        ("four_phi_F", 4, [F], 2),
        ("six_phi_F", 6, [F], 2),
        ("four_phi_H", 4, [H], 2),
        ("four_phi_K4", 4, [K4], 1),
        ("six_phi_K3", 6, [K3], 2),
    ]
    output = []
    for engine in ("symbolic", "numerical"):
        for name, q, vertices, order in cases:
            obs = [phi(chr(97 + k), f"x{k}") for k in range(q)]
            action = Action(vertices)
            values = {v.coupling: np.ones((2,) * v.n_fields) for v in vertices}
            setup_start = perf_counter()
            owner_maps = {
                n: _reference_owners(obs, vertices[0], n) for n in range(order + 1)
            }
            reference_owner_setup_ms = 1000 * (perf_counter() - setup_start)
            timings = {"late_connected_reference": [], "connected": []}
            counts = {}

            def generate(target):
                if target == "connected":
                    if engine == "symbolic":
                        return compute_moment(
                            obs, action, order, target=target
                        ).diagram_terms_by_order
                    return compute_moment_numerical(
                        obs, action, order, values, target=target
                    )
                if engine == "symbolic":
                    moment = baseline.compute_moment(
                        obs, action, order
                    ).diagram_terms_by_order
                else:
                    moment = baseline.compute_moment_numerical(
                        obs, action, order, values
                    )
                return {
                    n: [
                        dt
                        for dt in moment.get(n, [])
                        if _late_connected(dt, obs, vertices[0], n, owner_maps[n])
                    ]
                    for n in range(order + 1)
                }

            reference, early = generate("late_connected_reference"), generate(
                "connected"
            )
            exact_terms = 0
            signature = []
            for n in range(order + 1):
                assert len(reference.get(n, [])) == len(early.get(n, []))
                for old, new in zip(reference.get(n, []), early.get(n, [])):
                    for field in old.__dataclass_fields__:
                        assert getattr(old, field) == getattr(new, field), (
                            name,
                            engine,
                            field,
                        )
                    signature.append(
                        repr(
                            tuple(
                                getattr(old, field)
                                for field in old.__dataclass_fields__
                            )
                        )
                    )
                    fixed = {op.component_index: 0 for op in obs}
                    signature.append(
                        _array_receipt(
                            old.evaluate_coupling(values, fixed),
                            new.evaluate_coupling(values, fixed),
                        )
                    )
                    exact_terms += 1
            for target in timings:
                generate(
                    target
                )  # warm imports/nauty; timed generation still constructs every result
            for k in range(repeats):
                for target in (
                    ("late_connected_reference", "connected")
                    if k % 2 == 0
                    else ("connected", "late_connected_reference")
                ):
                    start = perf_counter()
                    dts = generate(target)
                    timings[target].append(1000 * (perf_counter() - start))
                    new_counts = {str(n): len(dts.get(n, [])) for n in range(order + 1)}
                    assert target not in counts or counts[target] == new_counts
                    counts[target] = new_counts
            medians = {target: median(values) for target, values in timings.items()}
            output.append(
                {
                    "case": name,
                    "engine": engine,
                    "n_components": 2,
                    "repeats": repeats,
                    "diagram_terms_by_order": counts,
                    "generation_median_ms": medians,
                    "timings_ms": timings,
                    "reference_owner_setup_ms_outside_timing": reference_owner_setup_ms,
                    "exact_ordered_term_and_array_matches": exact_terms,
                    "shared_output_metadata_and_arrays_sha256": hashlib.sha256(
                        "\n".join(signature).encode()
                    ).hexdigest(),
                    "reference_to_early_time_ratio": medians["late_connected_reference"]
                    / medians["connected"],
                }
            )
    controls = []
    for size in (2, 3):
        phi, psi = Field("phi", "physical", size), Field("psi", "response", size)
        action = Action(
            [Vertex([psi, phi, phi], "F"), Vertex([psi] * 3, "K", local=False)]
        )
        obs = [phi("a", "x"), phi("b", "y")]
        values = {
            "F": -1j * np.arange(size**3).reshape((size,) * 3) / 20,
            "K": 1j * np.arange(size**3).reshape((size,) * 3) / 60,
        }
        for engine in ("symbolic", "numerical"):

            def control(module):
                if engine == "symbolic":
                    return module.compute_moment(obs, action, 2).diagram_terms_by_order
                return module.compute_moment_numerical(obs, action, 2, values)

            before, after = control(baseline), control(candidate)
            signature = []
            for n in range(3):
                assert len(before.get(n, [])) == len(after.get(n, []))
                for old, new in zip(before.get(n, []), after.get(n, [])):
                    fields = tuple(old.__dataclass_fields__)
                    assert all(getattr(old, key) == getattr(new, key) for key in fields)
                    signature.append(
                        _array_receipt(
                            old.evaluate_coupling(values, {"a": 0, "b": 1}),
                            new.evaluate_coupling(values, {"a": 0, "b": 1}),
                        )
                    )
                    signature.append(repr(tuple(getattr(old, key) for key in fields)))
            timings = {"baseline": [], "candidate": []}
            for k in range(repeats):
                for label, module in (
                    (("baseline", baseline), ("candidate", candidate))
                    if k % 2 == 0
                    else (("candidate", candidate), ("baseline", baseline))
                ):
                    start = perf_counter()
                    control(module)
                    timings[label].append(1000 * (perf_counter() - start))
            medians = {label: median(times) for label, times in timings.items()}
            controls.append(
                {
                    "engine": engine,
                    "n_components": size,
                    "repeats": repeats,
                    "exact_ordered_term_and_array_matches": len(signature) // 2,
                    "shared_output_metadata_and_arrays_sha256": hashlib.sha256(
                        "\n".join(signature).encode()
                    ).hexdigest(),
                    "default_moment_median_ms": medians,
                    "timings_ms": timings,
                    "candidate_to_baseline_time_ratio": medians["candidate"]
                    / medians["baseline"],
                }
            )
    candidate_source = Path(candidate.__file__).resolve()
    helper_source = candidate_source.with_name("_connected.py")
    return {
        "cases": output,
        "default_moment_controls": controls,
        "candidate_source": str(candidate_source),
        "candidate_sha256": hashlib.sha256(candidate_source.read_bytes()).hexdigest(),
        "connected_helper_sha256": hashlib.sha256(
            helper_source.read_bytes()
        ).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "baseline_source": str(baseline_source),
        "baseline_sha256": hashlib.sha256(baseline_source.read_bytes()).hexdigest(),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "threads": {
                key: os.environ.get(key)
                for key in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "LOKY_MAX_CPU_COUNT",
                    "PYTHONHASHSEED",
                )
            },
        },
        "scope": "same connected target: frozen moment generation plus independent late incidence filter versus early target; metadata, couplings, weights and coefficient arrays agree exactly on bounded cases; no integration or convergence benchmark",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--baseline-source",
        type=Path,
        required=True,
        help="frozen preimplementation src/sft_wick/perturbation.py",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    result = run(args.repeats, args.baseline_source)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    for item in result["cases"]:
        print(
            item["engine"],
            item["case"],
            item["diagram_terms_by_order"],
            item["generation_median_ms"],
        )


if __name__ == "__main__":
    main()
