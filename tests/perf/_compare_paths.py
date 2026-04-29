"""Standalone comparison harness: runs the same workload with
the new vectorised path vs the per-sample scalar path and prints
both timings.  Not a pytest test -- just a developer's tool.

Usage::

    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        python tests/perf/_compare_paths.py
"""
from __future__ import annotations

import importlib.util as _ilu
import time
from pathlib import Path

import numpy as np

import sft_wick as sw
from sft_wick.evaluate import DynamicCouplingPromise, integrate_diagrams


_DEMO1_CLOSED_FORM_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples" / "demo1" / "c_closed_form.py"
)


def _load_demo1_C_fn():
    spec = _ilu.spec_from_file_location(
        "demo1_c_closed_form_cmp", _DEMO1_CLOSED_FORM_SRC,
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


def main() -> None:
    system = _make_dynamic_kappa3_system()
    expansion = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2])
    props = system.propagators(
        t_max=2.0, n_grid_t=20, c_closed_form=_load_demo1_C_fn(),
    )
    fk_dts = [
        dt for dt in expansion.dts_by_order[2]
        if expansion._vertex_type_label(dt) == "FK"
    ]

    coupling_values = system.build_coupling_values()
    fixed_indices = {"a": 0, "b": 1}

    n_runs = 50
    n_samples = 2 ** 12

    def run_once(seed: int):
        return integrate_diagrams(
            fk_dts,
            coupling_values=coupling_values,
            lambda_f=2.0,
            cache=props.cache,
            method="qmc_vectorized",
            n_samples=n_samples,
            seed=seed,
            positions={"x": 0.0, "y": 0.5},
            fixed_indices=fixed_indices,
            n_jobs=1,
        )

    # warmup
    run_once(0)

    # --- (a) vectorised default ---
    t0 = time.perf_counter()
    for run in range(n_runs):
        run_once(run)
    t_vec = time.perf_counter() - t0
    print(f"[vectorised] {n_runs} runs x n_samples={n_samples}: {t_vec:.3f}s")

    # --- (b) force scalar fallback by monkey-patching
    # ``DiagramTerm.evaluate_coupling_batched`` to raise. ---
    from sft_wick.perturbation import DiagramTerm

    real_batched = DiagramTerm.evaluate_coupling_batched

    def _raise(*args, **kwargs):
        raise NotImplementedError("forced scalar fallback")

    DiagramTerm.evaluate_coupling_batched = _raise  # type: ignore[assignment]
    try:
        # warmup
        run_once(0)
        t0 = time.perf_counter()
        for run in range(n_runs):
            run_once(run)
        t_scalar = time.perf_counter() - t0
        print(f"[scalar    ] {n_runs} runs x n_samples={n_samples}: {t_scalar:.3f}s")
    finally:
        DiagramTerm.evaluate_coupling_batched = real_batched  # type: ignore[assignment]

    print(f"\nspeedup: {t_scalar / t_vec:.2f}x")


if __name__ == "__main__":
    main()
