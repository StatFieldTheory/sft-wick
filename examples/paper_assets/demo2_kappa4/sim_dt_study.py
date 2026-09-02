#!/usr/bin/env python
"""Langevin simulation of demo2 (non-Gaussian noise, ``alpha``) at a chosen
time step, recording what the error budget needs.

Same physics and integrator as ``examples/demo2/run_simulation.py`` (Heun
step, AR(1) Ornstein-Uhlenbeck noise with a spatial Cholesky factor, the
quadratic deformation ``eta_tilde = eta + alpha (eta^2 - lam)``) but

* ``dt`` is a parameter, so the step-size bias can be measured by
  comparing runs;
* the fourth moment ``mu4_a(t) = <eta_tilde_a(0, t)^4>`` is recorded
  alongside ``mu2`` / ``mu3``, to cross-check the analytic ``kappa^(4)``;
* the per-realisation products are accumulated in sums AND sums of squares
  so the Monte-Carlo standard error of every ``xi_ab(r, t)`` is available.

Usage::

    python sim_dt_study.py --dt 0.05 --n_real 50000 --seed 7 --out sim_dt0.05.npz
"""
from __future__ import annotations

import argparse
import time

import numpy as np

lam, sigma_t, sigma_x, gamma = 0.05, 0.3, 1.0, 1.0
PAIRS = [(0, 0), (0, 1), (1, 1)]
R_VALUES = [0.0, 0.25, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
# Measurement times: the demo2 grid snapped to multiples of 0.02, so that
# runs at dt = 0.02 and dt = 0.01 measure at EXACTLY the same times (a
# 1.7% time mismatch at t = 0.6 is a 2.5% mismatch in xi there, larger
# than the step-size bias being measured).  dt = 0.05 runs land on the
# nearest step and record it in ``t_actual``.
T_NOMINAL = list(np.logspace(np.log10(0.1), np.log10(50.0), 15)) + [1.0, 5.0, 15.0]
T_MEASURE = sorted(set(round(round(t / 0.02) * 0.02, 10) for t in T_NOMINAL))


def _deform(eta, alpha):
    return eta if alpha == 0.0 else eta + alpha * (eta * eta - lam)


def simulate(dt_sim, n_real, seed, alpha, batch_size=500):
    rng = np.random.default_rng(seed)
    dx = sigma_x / 5.0
    n_sites = int(np.ceil(max(R_VALUES) / dx)) + 1
    x_grid = np.arange(n_sites) * dx
    n_steps = int(round((max(T_MEASURE) + 1.0) / dt_sim))
    K_space = np.exp(-np.abs(x_grid[:, None] - x_grid[None, :]) / sigma_x)
    L_space = np.linalg.cholesky(K_space)
    rho = np.exp(-dt_sim / sigma_t)
    sig_innov = np.sqrt(lam * (1 - rho ** 2))
    sig_init = np.sqrt(lam)
    # Measurement steps: nearest grid step to each requested time.
    t_idx = {t: int(round(t / dt_sim)) for t in T_MEASURE}
    t_actual = np.array([t_idx[t] * dt_sim for t in T_MEASURE])
    idx_to_t = {}
    for t, k in t_idx.items():
        idx_to_t.setdefault(k, []).append(t)

    n_t = len(T_MEASURE)
    ti = {t: i for i, t in enumerate(T_MEASURE)}
    xi_sum = np.zeros((3, n_t, n_sites))
    xi_sq = np.zeros((3, n_t, n_sites))
    mom = np.zeros((2, n_t, 3))          # mu2, mu3, mu4 sums at x=0
    n_good_t = np.zeros(n_t)

    n_batches = n_real // batch_size
    t0 = time.time()
    for bi in range(n_batches):
        z = [sig_init * (L_space @ rng.standard_normal((n_sites, batch_size)))
             for _ in range(2)]
        eta_k = [_deform(zi.T, alpha) for zi in z]
        phi = np.zeros((2, batch_size, n_sites))
        blown = np.zeros(batch_size, dtype=bool)
        for k in range(n_steps):
            z_next = [rho * zi + sig_innov * (L_space @ rng.standard_normal((n_sites, batch_size)))
                      for zi in z]
            eta_kn = [_deform(zi.T, alpha) for zi in z_next]
            f1 = -phi[0] + phi[1] ** 2 + eta_k[0]
            f2 = -phi[1] + phi[0] * phi[1] + eta_k[1]
            p1 = phi[0] + dt_sim * f1
            p2 = phi[1] + dt_sim * f2
            g1 = -p1 + p2 ** 2 + eta_kn[0]
            g2 = -p2 + p1 * p2 + eta_kn[1]
            phi[0] += 0.5 * dt_sim * (f1 + g1)
            phi[1] += 0.5 * dt_sim * (f2 + g2)
            blown |= np.any(np.abs(phi[0]) > 1e6, axis=1) | np.any(np.abs(phi[1]) > 1e6, axis=1)
            for t_m in idx_to_t.get(k + 1, ()):
                good = ~blown
                it = ti[t_m]
                n_good_t[it] += int(good.sum())
                for ip, (a, b) in enumerate(PAIRS):
                    prod = phi[a, good, 0:1] * phi[b, good]      # (n_good, n_sites)
                    xi_sum[ip, it] += prod.sum(axis=0)
                    xi_sq[ip, it] += (prod ** 2).sum(axis=0)
                for a in range(2):
                    e = eta_kn[a][good, 0]
                    mom[a, it, 0] += np.sum(e ** 2)
                    mom[a, it, 1] += np.sum(e ** 3)
                    mom[a, it, 2] += np.sum(e ** 4)
            eta_k, z = eta_kn, z_next
        if (bi + 1) % 20 == 0:
            print(f"  dt={dt_sim} batch {bi + 1}/{n_batches} ({time.time() - t0:.0f}s)", flush=True)

    n = n_good_t[:, None]
    mean = xi_sum / n                                   # (3, n_t, n_sites)
    var = xi_sq / n - mean ** 2
    sem = np.sqrt(np.maximum(var, 0.0) / n)
    r = np.array(R_VALUES)
    xi = np.stack([[np.interp(r, x_grid, mean[ip, it]) for it in range(n_t)] for ip in range(3)])
    xi_err = np.stack([[np.interp(r, x_grid, sem[ip, it]) for it in range(n_t)] for ip in range(3)])
    mu = mom / n_good_t[None, :, None]
    return dict(
        r=r, t=np.array(T_MEASURE), t_actual=t_actual, xi=xi, xi_err=xi_err,
        mu2=mu[:, :, 0], mu3=mu[:, :, 1], mu4=mu[:, :, 2],
        n_real=int(n_good_t.min()), dt_sim=dt_sim, seed=seed, alpha=alpha,
        lam=lam, sigma_t=sigma_t, sigma_x=sigma_x, pairs=np.array(PAIRS),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--n_real", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    res = simulate(a.dt, a.n_real, a.seed, a.alpha)
    np.savez(a.out, **res)
    print(f"wrote {a.out} in {time.time() - t0:.0f}s (n_real={res['n_real']})")


if __name__ == "__main__":
    main()
