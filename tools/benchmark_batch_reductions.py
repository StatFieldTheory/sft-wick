#!/usr/bin/env python3
"""Bounded source-pinned benchmark; timing and instrumentation are separate.

Run once for the frozen preimplementation source and once for the worktree.
Public cases use stable rotation+damping R and its analytic white-noise C.
Microcases expose isotropic/trace reuse and fully indexed negative controls.
Parity is not an independent scientific accuracy test.
"""

from __future__ import annotations

import argparse
import cProfile
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import tracemalloc
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve() / "src"))
    import numpy as np
    import sft_wick as sw
    from sft_wick import evaluate as ev
    from sft_wick.workflow.specs import ConstantImpulse

    assert (
        Path(ev.__file__).resolve()
        == (args.source / "src/sft_wick/evaluate.py").resolve()
    )

    callbacks = {"C": 0, "R": 0}
    count_callbacks = False

    @contextmanager
    def counts():
        nonlocal count_callbacks
        original_iso, original_einsum = ev._isotropic_C, np.einsum
        result = {"isotropic_C": 0, "trace": 0}

        def iso(*a, **kw):
            result["isotropic_C"] += 1
            return original_iso(*a, **kw)

        def einsum(spec, *a, **kw):
            if spec == "iaa->i":
                result["trace"] += 1
            return original_einsum(spec, *a, **kw)

        ev._isotropic_C, np.einsum = iso, einsum
        callbacks.update(C=0, R=0)
        count_callbacks = True
        try:
            yield result
        finally:
            count_callbacks = False
            result.update(callbacks)
            ev._isotropic_C, np.einsum = original_iso, original_einsum

    def measure(name, run):
        start = time.perf_counter()
        values = [np.asarray(run()).tolist()]
        walls = [time.perf_counter() - start]
        iterations = []
        for _ in range(6):
            start, count = time.perf_counter(), 0
            while time.perf_counter() - start < 0.025:
                value = run()
                count += 1
            walls.append((time.perf_counter() - start) / count)
            values.append(np.asarray(value).tolist())
            iterations.append(count)
        with counts() as operations:
            instrumented = np.asarray(run()).tolist()
        tracemalloc.start()
        run()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert all(value == values[0] for value in values + [instrumented])
        return {
            "name": name,
            "cold_seconds": walls[0],
            "warm_seconds": walls[1:],
            "warm_iterations": iterations,
            "warm_median_seconds": statistics.median(walls[1:]),
            "warm_min_seconds": min(walls[1:]),
            "warm_max_seconds": max(walls[1:]),
            "tracemalloc_peak_bytes": peak,
            "operations": operations,
            "values": values[0],
        }

    records = []
    profile_case = None
    for n in (1, 2, 4, 8):
        rng = np.random.default_rng(14)
        coeff = rng.normal(size=(n, n, n))
        dt = SimpleNamespace(
            propagator_indices=(("i", n), ("j", n), ("k", n)), n_external_response=0
        )
        ig = ev.DiagramIntegrand(dt, None, coeff)
        b = 256
        c = np.broadcast_to(np.eye(n), (b, n, n)).copy()
        r = np.broadcast_to(rng.normal(size=(n, n)), (b, n, n)).copy()
        rp, jac = np.ones(b), np.linspace(0.5, 1.0, b)
        for kind in ("iso", "trace", "indexed"):
            cb = (
                [(c, None, None)]
                if kind == "iso"
                else [(c, "i", "missing" if kind == "trace" else "j")]
            )
            rb = [(r, None if kind == "trace" else "j", "k", np.any(r != 0, axis=0))]
            for dynamic in (False, True):
                if dynamic:
                    samples = np.broadcast_to(coeff, (b,) + coeff.shape)

                    def run(ig=ig, samples=samples, cb=cb, rb=rb):
                        return ig._dynamic_values(samples, cb, rp, jac, r_batches=rb)

                else:

                    def run(ig=ig, cb=cb, rb=rb):
                        return ig._static_values(cb, rp, jac, r_batches=rb)

                records.append(
                    measure(
                        f"micro-N{n}-{kind}-{'dynamic' if dynamic else 'static'}", run
                    )
                )
                if n == 8 and kind == "iso" and not dynamic:
                    profile_case = run

    for n in (2, 4):
        # Orthogonal response blocks make C proportional to identity exactly.
        def response(t, s, n=n):
            if count_callbacks:
                callbacks["R"] += 1
            d = float(t - s)
            out = np.eye(n)
            for a in range(0, n, 2):
                angle = (a + 1) * 0.3 * d
                cs, sn = np.cos(angle), np.sin(angle)
                out[a : a + 2, a : a + 2] = [[cs, -sn], [sn, cs]]
            return np.exp(-d) * out if d >= 0 else np.zeros((n, n))

        def covariance(x, t, y, s, n=n):
            if count_callbacks:
                callbacks["C"] += 1
            t, s = np.broadcast_arrays(np.asarray(t), np.asarray(s))
            d, m = np.abs(t - s), np.minimum(t, s)
            # For unequal times, rotation must also be included. These public
            # cases have same-time C or explicitly retain its matrix indices.
            scalar = np.exp(-d) * (-np.expm1(-2 * m)) / 2
            out = scalar[..., None, None] * np.eye(n)
            for a in range(0, n, 2):
                angle = (a + 1) * 0.3 * (t - s)
                cs, sn = np.cos(angle), np.sin(angle)
                out[..., a, a] *= cs
                out[..., a + 1, a + 1] *= cs
                out[..., a, a + 1] = -scalar * sn
                out[..., a + 1, a] = scalar * sn
            return out

        system = sw.System(
            field=sw.FieldSpec("phi", n),
            linear=sw.ExplicitR(R_time=response, iso_R=False),
            vertices=[
                sw.LocalVertex(
                    "F", coupling=np.random.default_rng(5).normal(size=(n, n, n))
                )
            ],
            noise=sw.GaussianNoise(
                kappa2=sw.SeparableTranslation(
                    temporal=sw.ExponentialTemporal(lam=0.0, sigma_t=1.0),
                    spatial=sw.ExponentialSpatial(sigma_x=1.0),
                ),
                sigma2=ConstantImpulse(np.eye(n)),
            ),
        )
        props = system.propagators(
            t_max=1.1,
            c_closed_form=covariance,
            c_closed_form_only=True,
            c_closed_form_vectorized=True,
            diag_C=False,
            progress=False,
        )
        # Order-1 mean: its C is at the single vertex time, hence isotropic.
        for iso_c in (False, True):
            exp = system.expand(
                ("phi_a(x)",), orders=[1], iso_C=iso_c, diag_R=False, diag_C=False
            )

            def run(exp=exp):
                result = exp.evaluate(
                    props,
                    positions={"x": 0.0},
                    t_final=1.0,
                    component_pair=(0,),
                    orders=[1],
                    method="gauss_legendre",
                    n_gauss=32,
                )
                return [row["value"] for row in result.per_diagram]

            records.append(measure(f"public-N{n}-iso{iso_c}", run))

    profile = cProfile.Profile()
    profile.runcall(profile_case)
    stages = []
    import pstats

    for (file, line, name), (cc, nc, own, cumulative, callers) in pstats.Stats(
        profile
    ).stats.items():
        if name in {
            "_static_values",
            "_r_factor",
            "_c_product",
            "_isotropic_C",
            "_resolve_component",
            "_batch_reduction_memos",
        }:
            stages.append(
                {
                    "function": name,
                    "calls": nc,
                    "own_seconds": own,
                    "cumulative_seconds": cumulative,
                }
            )
    payload = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "source": str(args.source.resolve()),
        "loaded_evaluate_file": ev.__file__,
        "evaluate_sha256": hashlib.sha256(
            (args.source / "src/sft_wick/evaluate.py").read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "timing": "one cold call plus six >=25ms warm batches, per-call means; counters/profile/tracemalloc disabled in timings",
        "memory": "separate tracemalloc pass; not process RSS",
        "profile_micro_N8_iso_static": stages,
        "cases": records,
    }
    args.json.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cases": len(records),
                "source": payload["source"],
                "evaluate_sha256": payload["evaluate_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
