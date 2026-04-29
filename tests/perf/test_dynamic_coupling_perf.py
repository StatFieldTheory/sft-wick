"""Performance benchmark for the dynamic-coupling QMC path.

Locks an upper bound on the regression of the
``DynamicCouplingPromise.evaluate_at_batch`` codepath against the
pre-vectorisation baseline.  Marked ``slow`` -- run with::

    pytest tests/perf -m slow -s

The bench does ``N_RUNS`` calls to ``ig.integrate_moment_qmc_vectorized``
on the demo2 FK channel (the same channel profiled in the issue
that motivated the vectorisation).  The pre-vectorisation baseline
on an M-series Mac was ~70 s for ``N_RUNS=50, n_samples=4096``;
the post-vectorisation target is ``>= 10x`` speedup -- ``< 7 s``.
"""
from __future__ import annotations

import importlib.util as _ilu
import time
from pathlib import Path

import numpy as np
import pytest

import sft_wick as sw
from sft_wick.evaluate import integrate_diagrams


_DEMO1_CLOSED_FORM_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples" / "demo1" / "c_closed_form.py"
)


def _load_demo1_C_fn():
    spec = _ilu.spec_from_file_location(
        "demo1_c_closed_form_perf", _DEMO1_CLOSED_FORM_SRC,
    )
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.C_fn


def _make_dynamic_kappa3_system():
    N = 2
    F = np.zeros((N, N, N))
    F[0, 1, 1] = 1.0
    F[1, 0, 1] = 0.5
    F[1, 1, 0] = 0.5

    def kappa3_fn(n_list, t_list):
        n = np.asarray(n_list, dtype=float)
        t = np.asarray(t_list, dtype=float)
        envelope = float(
            np.exp(-abs(t[0] - t[1]) - abs(t[0] - t[2]))
            * np.exp(-abs(n[0] - n[1]) - abs(n[0] - n[2]))
        )
        K = np.zeros((N, N, N), dtype=complex)
        for a in range(N):
            K[a, a, a] = (1j / 6.0) * envelope
        return K

    return sw.System(
        field=sw.FieldSpec("phi", n_components=N),
        linear=sw.DiagonalA(gamma=[1.0, 1.0]),
        vertices=[sw.LocalVertex("F", coupling=F)],
        nonlocal_vertices=[
            sw.NonLocalVertex("K", order=3, coupling=kappa3_fn),
        ],
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
                spatial=sw.ExponentialSpatial(sigma_x=1.0),
            ),
        ),
    )


@pytest.mark.slow
def test_dynamic_coupling_evaluate_at_batch_speedup() -> None:
    """Benchmark: 50 calls to ``integrate_moment_qmc_vectorized`` on
    the demo2 FK channel with ``n_samples=4096``.

    Pre-vectorisation baseline (per the profiling at the top of the
    file): ~70 s wall time; >99% of which was inside
    ``_eval_symbolic`` per-sample.

    The post-vectorisation target is >= 10x speedup; we assert
    >= 5x as a generous floor that still flags any regression.
    """
    system = _make_dynamic_kappa3_system()
    expansion = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2])
    props = system.propagators(
        t_max=2.0, n_grid_t=20, c_closed_form=_load_demo1_C_fn(),
    )

    fk_dts = [
        dt for dt in expansion.dts_by_order[2]
        if expansion._vertex_type_label(dt) == "FK"
    ]
    assert fk_dts, "expected FK diagrams"

    coupling_values = system.build_coupling_values()
    fixed_indices = {"a": 0, "b": 1}

    # Warm up cache/JITs.
    integrate_diagrams(
        fk_dts,
        coupling_values=coupling_values,
        lambda_f=2.0,
        cache=props.cache,
        method="qmc_vectorized",
        n_samples=2 ** 12,
        seed=0,
        positions={"x": 0.0, "y": 0.5},
        fixed_indices=fixed_indices,
        n_jobs=1,
    )

    n_runs = 50
    n_samples = 2 ** 12  # 4096

    t0 = time.perf_counter()
    for run in range(n_runs):
        integrate_diagrams(
            fk_dts,
            coupling_values=coupling_values,
            lambda_f=2.0,
            cache=props.cache,
            method="qmc_vectorized",
            n_samples=n_samples,
            seed=run,
            positions={"x": 0.0, "y": 0.5},
            fixed_indices=fixed_indices,
            n_jobs=1,
        )
    elapsed = time.perf_counter() - t0
    print(
        f"\n[bench] dynamic-coupling QMC: "
        f"{n_runs} runs x n_samples={n_samples}: "
        f"{elapsed:.3f} s ({elapsed / n_runs * 1e3:.1f} ms / run)"
    )

    # Generous floor: pre-vectorisation was ~70 s on the same hw;
    # post-vectorisation target is < 14 s (5x).  The actual speedup
    # measured during development was ~30x.
    assert elapsed < 14.0, (
        f"perf regression: 50 demo2-FK runs took {elapsed:.1f}s "
        f"(expected < 14s after _eval_symbolic_batched landed)."
    )
