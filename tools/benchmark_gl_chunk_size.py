#!/usr/bin/env python3
"""Wall time and peak RSS of one GL evaluation for several ``gl_chunk_size`` bounds.

Run with the sft-wick conda Python::

    python tools/benchmark_gl_chunk_size.py --order 2 --n-gauss 24
    python tools/benchmark_gl_chunk_size.py --order 4 --n-gauss 7

Each bound runs in its own subprocess with one BLAS/OpenMP thread, so the
peak resident set size of one bound is not inherited by the next.  The model
is a two-component field with a cubic local vertex, unequal decay rates and
an exponential separable noise kernel; both external times are integrated,
so an order-``n`` diagram has ``n + 2`` time variables.  The table in
``docs/user_guide/workflow.rst`` (``gl-batch-memory``) was produced with the
two commands above.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

THREAD_ENV = {
    "SFT_WICK_PROGRESS": "0",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def _bound(text: str) -> int | None:
    return None if text == "eager" else int(text)


def worker(order: int, n_gauss: int, bound: int | None) -> dict:
    import numpy as np
    import sft_wick as sw

    rng = np.random.default_rng(3)
    system = sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=[0.7, 1.6]),
        vertices=[sw.LocalVertex("F", coupling=rng.normal(size=(2, 2, 2)))],
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.3, sigma_t=0.4),
                spatial=sw.ExponentialSpatial(sigma_x=1.0),
            )
        ),
        t_min=0.2,
    )
    expansion = system.expand(("phi_a(x)", "phi_b(y)"), orders=[order], progress=False)
    props = system.propagators(t_max=3.0, n_grid_t=40, progress=False)
    start = time.perf_counter()
    result = expansion.evaluate(
        props,
        positions={"x": 0.0, "y": 0.6},
        t_final=2.5,
        component_pair=(0, 1),
        integrate_over="all",
        orders=[order],
        method="gauss_legendre",
        n_gauss=n_gauss,
        gl_chunk_size=bound,
    )
    seconds = time.perf_counter() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / 2**20 if sys.platform == "darwin" else rss / 2**10
    return {
        "order": order,
        "n_gauss": n_gauss,
        "gl_chunk_size": bound,
        "seconds": round(seconds, 2),
        "peak_rss_MB": round(rss_mb),
        "total": repr(complex(result.total)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--n-gauss", type=int, default=24)
    parser.add_argument(
        "--bounds",
        nargs="+",
        default=["eager", "65536", "4096"],
        help="gl_chunk_size values; 'eager' means None.",
    )
    parser.add_argument("--json", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.order, args.n_gauss, _bound(args.bounds[0]))))
        return
    rows = []
    for bound in args.bounds:
        out = subprocess.run(
            [sys.executable, __file__, "--worker", "--order", str(args.order),
             "--n-gauss", str(args.n_gauss), "--bounds", bound],
            env={**os.environ, **THREAD_ENV},
            check=True,
            capture_output=True,
            text=True,
        )
        rows.append(json.loads(out.stdout.strip().splitlines()[-1]))
        print(json.dumps(rows[-1]))
    if args.json is not None:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
