#!/usr/bin/env python3
"""Coupling-read timings for the source tree given by ``--source``, including full expansion preparation."""

from __future__ import annotations

import argparse
import cProfile
import gc
import hashlib
import json
from pathlib import Path
import platform
import pstats
import statistics
import sys
import time
import tracemalloc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--disable-probe", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve() / "src"))
    import numpy as np
    import sft_wick as sw
    from sft_wick import perturbation as pt
    from sft_wick.expressions import Product, Propagator, Rational, Sum, Symbol

    assert (
        Path(pt.__file__).resolve()
        == (args.source / "src/sft_wick/perturbation.py").resolve()
    )
    if args.disable_probe:
        import sft_wick._coupling_probe as probe

        probe.known_scalar_shape = lambda *args: None
    callbacks = {"K": 0, "C": 0}
    recording = False

    def encode(value):
        a = np.asarray(value)
        return {
            "dtype": str(a.dtype),
            "shape": list(a.shape),
            "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            "real": a.real.tolist(),
            "imag": a.imag.tolist(),
        }

    def measure(label, run):
        nonlocal recording
        start = time.perf_counter()
        cold = run()
        cold_seconds = time.perf_counter() - start
        reference = encode(cold)
        walls, counts = [], []
        for _ in range(6):
            start, count = time.perf_counter(), 0
            while time.perf_counter() - start < 0.025 or count < 3:
                value = run()
                count += 1
            walls.append((time.perf_counter() - start) / count)
            counts.append(count)
            assert encode(value) == reference
        callbacks.update(K=0, C=0)
        recording = True
        value = run()
        recording = False
        assert encode(value) == reference
        operation_counts = dict(callbacks)
        tracemalloc.start()
        run()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "name": label,
            "cold_seconds": cold_seconds,
            "warm_seconds": walls,
            "iterations": counts,
            "warm_median_seconds": statistics.median(walls),
            "callbacks": operation_counts,
            "tracemalloc_peak_bytes": peak,
            "value": reference,
        }

    cases = []
    for n in (1, 2, 4):
        for count in (1, 24):
            expr = Sum(
                tuple(
                    Product((Rational(1, 3), Symbol("A", ("i",)), Symbol("B", ("j",))))
                    for _ in range(count)
                )
            )
            for batch in (None, 128):
                dt = pt.DiagramTerm(
                    (Propagator("C", "i", "j", "x", "y"),),
                    expr,
                    Rational(2, 3),
                    (),
                    (("i", n), ("j", n)),
                    0,
                )
                cv = {
                    "A": np.arange(n, dtype=float) + 0.125,
                    "B": np.arange(n, dtype=float) + 0.75,
                }
                if batch is None:

                    def run(dt=dt, cv=cv):
                        return dt.evaluate_coupling(cv)

                else:
                    cv["A"] = cv["A"][None] + np.arange(batch)[:, None] / 128

                    def run(dt=dt, cv=cv):
                        return dt.evaluate_coupling_batched(cv, 128)

                cases.append(measure(f"coupling-N{n}-terms{count}-B{batch or 0}", run))
    for batch in (None, 128):
        dt = pt.DiagramTerm((), Symbol("A"), Rational(1), (), (), 0)
        cv = {"A": np.array(0.2) if batch is None else np.full(128, 0.2)}
        run = (
            (lambda: dt.evaluate_coupling(cv))
            if batch is None
            else (lambda: dt.evaluate_coupling_batched(cv, 128))
        )
        cases.append(measure(f"leaf-B{batch or 0}", run))

    setup_start = time.perf_counter()
    n = 2
    f = np.random.default_rng(11).normal(size=(n, n, n)) * 0.1
    k = np.full((n, n, n), 0.003)

    def coupling(positions, times):
        if recording:
            callbacks["K"] += 1
        scale = 1 + 0.05 * np.sum(times, axis=0)
        return scale[:, None, None, None] * k

    def covariance(x, t, y, s):
        if recording:
            callbacks["C"] += 1
        t, s = np.broadcast_arrays(np.asarray(t), np.asarray(s))
        scale = np.exp(-np.abs(t - s)) * (-np.expm1(-2 * np.minimum(t, s))) / 2
        return scale[..., None, None] * np.eye(n)

    system = sw.System(
        field=sw.FieldSpec("phi", n),
        linear=sw.DiagonalA(gamma=[1.0, 1.0]),
        vertices=[sw.LocalVertex("F", coupling=f)],
        nonlocal_vertices=[
            sw.NonLocalVertex(
                "K",
                order=3,
                coupling=coupling,
                coupling_vectorized=True,
                already_R_contracted=True,
            )
        ],
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.0, sigma_t=1.0),
                spatial=sw.ExponentialSpatial(sigma_x=1.0),
            )
        ),
    )
    expansion = system.expand(("phi_a(x)", "phi_b(y)"), orders=[4])
    fk = [
        dt
        for dt in expansion.dts_by_order[4]
        if expansion._vertex_type_label(dt) == "FK" and dt.propagator_indices
    ]
    cv = system.build_coupling_values()

    def tree_size(expr):
        if isinstance(expr, Product):
            return 1 + sum(tree_size(x) for x in expr.factors)
        if isinstance(expr, Sum):
            return 1 + sum(tree_size(x) for x in expr.terms)
        return 1

    dt = max(fk, key=lambda term: tree_size(term.coupling_sum))
    setup_seconds = time.perf_counter() - setup_start
    profile_run = None
    for batch in (1, 128, 512):
        ig = dt.build_integrand(cv, fixed_indices={"a": 0, "b": 1})
        promise = ig.dynamic_coupling
        labels = {x for legs in promise.spatial_args_by_name.values() for x in legs}
        label_t = {x: np.linspace(0.1, 0.8, batch) for x in labels}
        label_x = {x: 0.0 for x in labels}

        def run(p=promise, t=label_t, x=label_x, b=batch):
            return p.evaluate_at_batch(t, x, b)

        cases.append(measure(f"public-F3K-coupling-B{batch}", run))
        if batch == 128:
            profile_run = run
    props = system.propagators(
        t_max=1.1,
        c_closed_form=covariance,
        c_closed_form_only=True,
        c_closed_form_vectorized=True,
        progress=False,
    )
    for chunk in (None, 16):
        ig = dt.build_integrand(cv, fixed_indices={"a": 0, "b": 1})

        def run(ig=ig, chunk=chunk):
            return ig.integrate_moment_gauss_legendre(
                cache=props.cache,
                lambda_f=1.0,
                positions={"x": 0.0, "y": 0.5},
                n_gauss=4,
                gl_chunk_size=chunk,
            )[0]

        cases.append(measure(f"public-F3K-GL-chunk{chunk or 0}", run))
    profile = cProfile.Profile()
    profile.runcall(profile_run)
    stages = []
    for (_, _, name), (_, calls, own, cumulative, _) in pstats.Stats(
        profile
    ).stats.items():
        if name in (
            "_eval_symbolic",
            "_eval_symbolic_batched",
            "scalar",
            "batched",
            "_scalar_symbol",
            "_batched_symbol",
            "_indices",
            "evaluate_at_batch",
            "evaluate_coupling",
            "evaluate_coupling_batched",
            "bind_indices",
            "known_scalar_shape",
        ):
            stages.append(
                {
                    "name": name,
                    "calls": calls,
                    "self_seconds": own,
                    "cumulative_seconds": cumulative,
                }
            )
    metadata_allocation = None
    try:
        from sft_wick._coupling_plan import compile_coupling_plan
    except ImportError:
        pass
    else:
        import sft_wick._coupling_plan as cp

        shared = getattr(cp, "_shared_core", None)
        if shared:
            shared.cache_clear()
        tracemalloc.start()
        metadata_plan = compile_coupling_plan(dt.coupling_sum)
        gc.collect()
        retained, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        metadata_allocation = {
            "node_count": metadata_plan.node_count if metadata_plan else None,
            "retained_bytes": retained,
            "peak_bytes": peak,
        }
    # Compilation only; input copies prepared outside measured windows.
    import copy
    from sft_wick._coupling_plan import compile_coupling_plan
    import sft_wick._coupling_plan as cp

    shared = getattr(cp, "_shared_core", None)
    preparation = {}
    for label in ("duplicate", "unique"):
        expressions = (
            [copy.deepcopy(dt.coupling_sum) for _ in range(24)]
            if label == "duplicate"
            else [
                Sum(
                    tuple(
                        Product(
                            (
                                Rational(1, 3),
                                Symbol(f"A{i}", ("j",)),
                                Symbol("B", ("k",)),
                            )
                        )
                        for _ in range(24)
                    )
                )
                for i in range(24)
            ]
        )
        walls = []
        for _ in range(5):
            if shared:
                shared.cache_clear()
            gc.collect()
            start = time.perf_counter()
            plans = [compile_coupling_plan(expr) for expr in expressions]
            walls.append(time.perf_counter() - start)
            assert all(p is not None for p in plans)
            del plans
        if shared:
            shared.cache_clear()
        gc.collect()
        tracemalloc.start()
        plans = [compile_coupling_plan(expr) for expr in expressions]
        gc.collect()
        retained, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        preparation[label] = {
            "plans": len(plans),
            "seconds": walls,
            "median_seconds": statistics.median(walls),
            "retained_bytes": retained,
            "peak_bytes": peak,
            "distinct_cores": len({id(getattr(p, "core", p)) for p in plans}),
            "cache_info": shared.cache_info()._asdict() if shared else None,
        }
        del plans
    # Whole real expansion, preserving each expression and each diagram.
    expressions = [term.coupling_sum for term in expansion.dts_by_order[4]]
    walls = []
    for _ in range(5):
        if shared:
            shared.cache_clear()
        gc.collect()
        start = time.perf_counter()
        plans = [compile_coupling_plan(expr) for expr in expressions]
        walls.append(time.perf_counter() - start)
        del plans
    if shared:
        shared.cache_clear()
    gc.collect()
    tracemalloc.start()
    plans = [compile_coupling_plan(expr) for expr in expressions]
    gc.collect()
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    admitted = [plan for plan in plans if plan is not None]
    preparation["real_order4_expansion"] = {
        "terms": len(plans),
        "admitted": len(admitted),
        "seconds": walls,
        "median_seconds": statistics.median(walls),
        "retained_bytes": retained,
        "peak_bytes": peak,
        "distinct_cores": len({id(plan.core) for plan in admitted}),
        "cache_info": shared.cache_info()._asdict() if shared else None,
    }
    del admitted, plans
    # Dynamic scalar-leaf negative control, callbacks counted as K.
    from sft_wick.evaluate import DynamicCouplingPromise

    def tiny_kernel(x, t):
        if recording:
            callbacks["K"] += 1
        return np.full(t.shape[1], 0.2)

    tiny_kernel.vectorized = True
    tiny = DynamicCouplingPromise(
        pt.DiagramTerm((), Symbol("A"), Rational(1), (), (), 0),
        {},
        {"A": tiny_kernel},
        {"A": ("x",)},
        {},
    )
    cases.append(
        measure(
            "dynamic-leaf-B128",
            lambda: tiny.evaluate_at_batch(
                {"x": np.linspace(0, 1, 128)}, {"x": 0.0}, 128
            ),
        )
    )
    symbol_tuples = []

    def collect_symbols(expr):
        if type(expr) is Symbol:
            symbol_tuples.append(expr.indices)
        elif type(expr) is Product:
            for child in expr.factors:
                collect_symbols(child)
        elif type(expr) is Sum:
            for child in expr.terms:
                collect_symbols(child)

    collect_symbols(dt.coupling_sum)
    source_hashes = {
        str(p.relative_to(args.source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((args.source / "src").rglob("*.py"))
    }
    payload = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "source": str(args.source.resolve()),
        "source_sha256": source_hashes,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "setup_seconds": setup_seconds,
        "order4_diagrams": len(expansion.dts_by_order[4]),
        "prop_indexed_fk_diagrams": len(fk),
        "selected_term": dt.to_latex(),
        "selected_expression_nodes": tree_size(dt.coupling_sum),
        "selected_symbol_leaves": len(symbol_tuples),
        "selected_unique_index_tuples": len(set(symbol_tuples)),
        "cases": cases,
        "profile": stages,
        "metadata_allocation": metadata_allocation,
        "preparation": preparation,
        "disable_probe": args.disable_probe,
    }
    args.json.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.json),
                "cases": len(cases),
                "setup_seconds": setup_seconds,
            }
        )
    )


if __name__ == "__main__":
    main()
