#!/usr/bin/env python
"""Error budget for demo2's two-point functions: theory channels vs
Langevin simulation, with the ingredients the paper needs.

Theory (all through the L1 API on the current package), on the time grid
of the simulations (the demo2 grid snapped to multiples of 0.02):

* ``0``     order 0 with the EXACT two-kernel effective covariance
            ``kappa2_eff = lam k + 2 alpha^2 lam^2 k^2`` (both pieces are
            separable-exponential, so both have the built-in closed form;
            their sum is passed as ``c_closed_form``), and for comparison
            the single-kernel ``lam_eff = lam (1 + 2 alpha^2 lam)``
            approximation used by ``examples/demo2/L2`` so far;
* ``FF``    order 2, same two C variants;
* ``FK``    order 2, F x kappa^(3), through the R-CONTRACTED kernel
            (``examples/demo2/k3_R_coupling.py``): the three leg integrals
            over the narrow kernel are done analytically/with a composite
            rule inside the callable, the outer integral is 1-D and a
            32-node Gauss-Legendre rule is converged to 1e-4.  Also the
            raw-kernel, 4-D tensor rule (n = 8) the paper used, to show
            its error;
* ``FFK4``  order 3, F x F x kappa^(4) -- the leading kappa^(4)
            contribution to <phi phi> (F x kappa^(4) at order 2 vanishes:
            2 + n_F - 4 n_K4 must be a non-negative even number) -- via the
            R-contracted ``k4_R_contracted.py``; three pure-R^6 diagrams,
            2-D outer integral.  Identically zero for xi_01 (kappa^(4) is
            even under phi_1 -> -phi_1);
* ``FFFF``  order 4 (64 diagrams), exact C_eff;
* ``FFFK``  order 4, F^3 x kappa^(3) -- cannot go through the
            dynamic-coupling path (a kappa^(3) index sits on a C
            propagator), so it is estimated with kappa^(3) collapsed to an
            equal-time constant, calibrated on FK at order 2 where the
            exact answer is known.

Simulation: ``sim_dt_study.py`` at dt = 0.02 and 0.01, 20 seeds x 100k
realisations each (2M per step size), measured at exactly the theory
times, Richardson-extrapolated to dt -> 0 assuming the Heun O(dt^2)
bias; plus the shipped 200k cache (dt = 0.05, nominal times) for
reference.

Outputs: ``budget.npz`` (every channel on the grid), ``budget_meta.json``;
``make_figures.py`` turns them into ``budget.md`` and the figures.
Stages are cached in ``cache2/`` so a re-run only recomputes what is
missing.  Run (a few minutes on many cores)::

    OMP_NUM_THREADS=1 python run_budget.py
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "demo2"))

import sft_wick as sw  # noqa: E402
from sft_wick.workflow.closed_forms import ClosedFormC  # noqa: E402
from k3_coupling import coupling_fn_vectorized as k3_raw_fn  # noqa: E402
from k3_R_coupling import coupling_fn_vectorized as k3_R_fn  # noqa: E402
from k4_R_contracted import coupling_fn_vectorized as k4_R_fn  # noqa: E402
from sim_dt_study import T_MEASURE  # noqa: E402

LAM, SIGMA_T, SIGMA_X, GAMMA, ALPHA, N = 0.05, 0.3, 1.0, 1.0, 0.6, 2
LAM_EFF = LAM * (1 + 2 * ALPHA ** 2 * LAM)
T_MAX = 50.0
PAIRS = [(0, 0), (0, 1), (1, 1)]
CACHE = HERE / "cache2"
CACHE.mkdir(exist_ok=True)
N_JOBS = -1

F = np.zeros((N, N, N))
F[0, 1, 1] = 1.0
F[1, 0, 1] = F[1, 1, 0] = 0.5

_CF_A = ClosedFormC(gamma=(GAMMA, GAMMA), lam=LAM, sigma_t=SIGMA_T,
                    spatial=sw.ExponentialSpatial(sigma_x=SIGMA_X))
_CF_B = ClosedFormC(gamma=(GAMMA, GAMMA), lam=2 * ALPHA ** 2 * LAM ** 2,
                    sigma_t=SIGMA_T / 2, spatial=sw.ExponentialSpatial(sigma_x=SIGMA_X / 2))


def C_eff_exact(n1, t1, n2, t2):
    return _CF_A(n1, t1, n2, t2) + _CF_B(n1, t1, n2, t2)


K3_EQ = np.zeros((N, N, N))
for _a in range(N):
    K3_EQ[_a, _a, _a] = 24.0 * ALPHA * LAM ** 2 * SIGMA_T ** 2


def make_system(temporal_lam, vertex=None):
    return sw.System(
        field=sw.FieldSpec("phi", n_components=N),
        linear=sw.DiagonalA(gamma=[GAMMA, GAMMA]),
        vertices=[sw.LocalVertex("F", coupling=F)],
        nonlocal_vertices=[vertex] if vertex is not None else [],
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=temporal_lam, sigma_t=SIGMA_T),
            spatial=sw.ExponentialSpatial(sigma_x=SIGMA_X))),
    )


def props_for(system, exact):
    kw = dict(t_max=T_MAX, n_grid_t=60, c_closed_form_only=True,
              c_closed_form_vectorized=True, progress=False)
    if exact:
        return system.propagators(c_closed_form=C_eff_exact, **kw)
    return system.propagators(c_closed_form="auto", **kw)


def sweep_rows(system, props, orders, vertex_types, r_list, t_list, pairs,
               method, n_samples=32768, n_gauss=8, label=""):
    expansion = system.expand(("phi_a(x)", "phi_b(y)"), orders=orders, progress=False)
    t0 = time.perf_counter()
    res = expansion.sweep(
        props, positions_grid={"x": [0.0], "y": list(r_list)},
        t_final_grid=list(t_list), component_pairs=pairs, orders=orders,
        vertex_types=vertex_types, method=method, n_samples=n_samples,
        n_gauss=n_gauss, n_jobs=N_JOBS, progress=True,
    )
    df = pd.DataFrame(res.rows)
    out = (df.groupby(["y", "t_final", "a", "b", "vertex_type", "order"], as_index=False)
             ["value"].sum())
    out["seconds"] = time.perf_counter() - t0
    print(f"[{label}] {len(df)} diagram rows in {time.perf_counter() - t0:.1f}s", flush=True)
    return out


def cached(name, fn):
    path = CACHE / f"{name}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    df = fn()
    df.to_pickle(path)
    return df


def load_sims(dt):
    files = sorted(glob.glob(str(HERE / "sims" / f"sim_dt{dt}_s*.npz")))
    xs, es, n_tot, mus = [], [], 0, []
    t_ref = None
    for f in files:
        d = np.load(f)
        if t_ref is None:
            t_ref = d["t"]
            assert np.allclose(d["t_actual"], d["t"]), "measurement times not on the dt grid"
        xs.append(d["xi"]); es.append(d["xi_err"]); n_tot += int(d["n_real"])
        mus.append(np.stack([d["mu2"], d["mu3"], d["mu4"]]))
    xs, es = np.array(xs), np.array(es)
    w = 1.0 / es ** 2
    mean = np.sum(w * xs, axis=0) / np.sum(w, axis=0)
    err = 1.0 / np.sqrt(np.sum(w, axis=0))
    return dict(t=t_ref, xi=mean, err=err, n_real=n_tot, n_files=len(files),
                mu=np.mean(mus, axis=0))


def main():
    sim200 = {k: v for k, v in np.load(HERE.parents[1] / "demo2" / "sim_cache_a0.6.npz").items()}
    sims = {dt: load_sims(dt) for dt in ("0.02", "0.01")}
    t_list = [float(t) for t in T_MEASURE]
    assert np.allclose(sims["0.02"]["t"], t_list) and np.allclose(sims["0.01"]["t"], t_list)
    r_list = [float(r) for r in sim200["r"]]
    r_sub = [0.0, 0.5]
    t_check = [1.0, 3.48, 15.0]

    sys_eff = make_system(LAM_EFF)
    sys_bare = make_system(LAM)
    sys_k3raw = make_system(LAM, sw.NonLocalVertex("K", 3, coupling=k3_raw_fn, coupling_vectorized=True))
    sys_k3R = make_system(LAM, sw.NonLocalVertex("K", 3, coupling=k3_R_fn, coupling_vectorized=True,
                                                 already_R_contracted=True))
    sys_k4R = make_system(LAM, sw.NonLocalVertex("K4", 4, coupling=k4_R_fn, coupling_vectorized=True,
                                                 already_R_contracted=True))
    sys_k3eq = make_system(LAM, sw.NonLocalVertex("K", 3, coupling=K3_EQ, equal_time=True))

    ff_exact = cached("ff_exact", lambda: sweep_rows(
        sys_bare, props_for(sys_bare, True), [0, 2], None, r_list, t_list, PAIRS,
        "qmc_vectorized", label="0+FF exact C_eff"))
    ff_lameff = cached("ff_lameff", lambda: sweep_rows(
        sys_eff, props_for(sys_eff, False), [0, 2], None, r_list, t_list, PAIRS,
        "qmc_vectorized", label="0+FF lam_eff"))
    fk_rc = cached("fk_rc", lambda: sweep_rows(
        sys_k3R, props_for(sys_k3R, False), [2], ["FK"], r_list, t_list, PAIRS,
        "gauss_legendre", n_gauss=32, label="FK R-contracted GL32"))
    fk_rc64 = cached("fk_rc64", lambda: sweep_rows(
        sys_k3R, props_for(sys_k3R, False), [2], ["FK"], r_sub, t_list, [(0, 1)],
        "gauss_legendre", n_gauss=64, label="FK R-contracted GL64 (check)"))
    fk_raw8 = cached("fk_raw8", lambda: sweep_rows(
        sys_k3raw, props_for(sys_k3raw, False), [2], ["FK"], r_sub, t_list, [(0, 1)],
        "gauss_legendre", n_gauss=8, label="FK raw kernel GL8 (as in the paper)"))
    ffk4_rc = cached("ffk4_rc", lambda: sweep_rows(
        sys_k4R, props_for(sys_k4R, False), [3], ["FK4"], r_sub, t_list, PAIRS,
        "gauss_legendre", n_gauss=12, label="FFK4 R-contracted GL12"))
    ffk4_rc16 = cached("ffk4_rc16", lambda: sweep_rows(
        sys_k4R, props_for(sys_k4R, False), [3], ["FK4"], [0.0], t_check, [(0, 0)],
        "gauss_legendre", n_gauss=16, label="FFK4 R-contracted GL16 (check)"))
    ffff = cached("ffff", lambda: sweep_rows(
        sys_bare, props_for(sys_bare, True), [4], None, r_sub, t_list, PAIRS,
        "qmc_vectorized", label="FFFF exact C_eff"))
    fk_eq = cached("fk_eq", lambda: sweep_rows(
        sys_k3eq, props_for(sys_k3eq, False), [2], ["FK"], [0.0], t_check, [(0, 1)],
        "qmc_vectorized", label="FK equal-time-constant (calibration)"))
    fffk_eq = cached("fffk_eq", lambda: sweep_rows(
        sys_k3eq, props_for(sys_k3eq, False), [4], ["FK"], [0.0], t_check, [(0, 1)],
        "qmc_vectorized", label="FFFK equal-time-constant (magnitude)"))

    def grid(df, vt, order, pair, r_vals, t_vals):
        sub = df[(df.vertex_type == vt) & (df.order == order)
                 & (df.a == pair[0]) & (df.b == pair[1])]
        g = np.full((len(t_vals), len(r_vals)), np.nan)
        for _, row in sub.iterrows():
            ti = int(np.argmin(np.abs(np.asarray(t_vals) - row.t_final)))
            ri = int(np.argmin(np.abs(np.asarray(r_vals) - row.y)))
            g[ti, ri] = row.value
        return g

    out = {"t": np.array(t_list), "r": np.array(r_list), "r_sub": np.array(r_sub),
           "t_check": np.array(t_check)}
    for pair in PAIRS:
        key = f"{pair[0]}{pair[1]}"
        out[f"o0_exact_{key}"] = grid(ff_exact, "", 0, pair, r_list, t_list)
        out[f"ff_exact_{key}"] = grid(ff_exact, "F", 2, pair, r_list, t_list)
        out[f"o0_lameff_{key}"] = grid(ff_lameff, "", 0, pair, r_list, t_list)
        out[f"ff_lameff_{key}"] = grid(ff_lameff, "F", 2, pair, r_list, t_list)
        out[f"fk_{key}"] = grid(fk_rc, "FK", 2, pair, r_list, t_list)
        out[f"ffk4_{key}"] = grid(ffk4_rc, "FK4", 3, pair, r_sub, t_list)
        out[f"ffff_{key}"] = grid(ffff, "F", 4, pair, r_sub, t_list)
    out["fk64_01"] = grid(fk_rc64, "FK", 2, (0, 1), r_sub, t_list)
    out["fk_raw8_01"] = grid(fk_raw8, "FK", 2, (0, 1), r_sub, t_list)
    out["ffk4_16_00"] = grid(ffk4_rc16, "FK4", 3, (0, 0), [0.0], t_check)
    out["fk_eq_01"] = grid(fk_eq, "FK", 2, (0, 1), [0.0], t_check)
    out["fffk_eq_01"] = grid(fffk_eq, "FK", 4, (0, 1), [0.0], t_check)

    out["sim200_t"] = sim200["t"]
    out["sim200_xi"] = sim200["xi"]
    for dt, s in sims.items():
        out[f"sim_dt{dt}_xi"] = s["xi"]
        out[f"sim_dt{dt}_err"] = s["err"]
        out[f"sim_dt{dt}_mu"] = s["mu"]            # (3, 2, n_t): mu2, mu3, mu4
    x2, x1 = out["sim_dt0.02_xi"], out["sim_dt0.01_xi"]
    e2, e1 = out["sim_dt0.02_err"], out["sim_dt0.01_err"]
    out["sim_extrap_xi"] = (4 * x1 - x2) / 3
    out["sim_extrap_err"] = np.sqrt((4 * e1) ** 2 + e2 ** 2) / 3
    np.savez(HERE / "budget.npz", **out)
    meta = dict(lam=LAM, sigma_t=SIGMA_T, sigma_x=SIGMA_X, gamma=GAMMA, alpha=ALPHA,
                lam_eff=LAM_EFF, n_real_cache=int(sim200["n_real"]), dt_cache=float(sim200["dt_sim"]),
                n_real_sims={dt: s["n_real"] for dt, s in sims.items()},
                n_files_sims={dt: s["n_files"] for dt, s in sims.items()},
                seconds={name: float(df.seconds.iloc[0]) for name, df in [
                    ("ff_exact", ff_exact), ("ff_lameff", ff_lameff), ("fk_rc", fk_rc),
                    ("fk_rc64", fk_rc64), ("fk_raw8", fk_raw8), ("ffk4_rc", ffk4_rc),
                    ("ffk4_rc16", ffk4_rc16), ("ffff", ffff), ("fk_eq", fk_eq),
                    ("fffk_eq", fffk_eq)]})
    (HERE / "budget_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
