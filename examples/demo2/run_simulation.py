"""Run Langevin simulation with NON-GAUSSIAN driving noise.

Variant of demo1/run_simulation.py. The only physical change is a quadratic
deformation of the driving noise:

    eta_tilde_a(x, t) = eta_a(x, t) + alpha * (eta_a(x, t)**2 - lam)

where eta_a is the same Gaussian AR(1)+spatial-Cholesky field used in demo1
(stationary variance lam per component per site). The deformation introduces
a non-zero 3rd-order cumulant while leaving the dominant part of the 2-point
cumulant intact. At alpha=0 the output reduces exactly to demo1.

Produces: sim_cache.npz containing
  - xi_ab(r, t) for all component pairs (11, 12, 22)  [shape (3, n_t, n_r)]
  - var0_ab(t) per pair                                [shape (3, n_t)]
  - mu3_a(t) = <eta_tilde_a(0, t)**3>                  [shape (2, n_t)]
  - mu2_a(t) = <eta_tilde_a(0, t)**2>                  [shape (2, n_t)]
The mu2/mu3 arrays let the analysis notebook cross-check the analytical
kappa2_eff and kappa3 against the simulator.

Usage:
    python run_simulation.py [--n_real 50000] [--dt 0.02] [--seed 123]
                             [--alpha 0.3]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

# ---------- Physical parameters (must match notebook) ----------
lam: float = 0.05
sigma_t: float = 0.3
sigma_x: float = 1.0
gamma: float = 1.0
N_comp: int = 2
T_MAX: float = 50.0

PAIRS: list[tuple[int, int]] = [(0, 0), (0, 1), (1, 1)]


def simulate_correlators(
    r_values: list[float],
    t_measure_list: list[float],
    n_real: int = 50_000,
    dt_sim: float = 0.02,
    seed: int = 123,
    alpha: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Heun (RK2) simulation with AR(1) noise + quadratic non-Gaussian deformation.

    Parameters
    ----------
    r_values : spatial separations at which xi_ab(r, t) is measured.
    t_measure_list : measurement times.
    n_real : total number of realisations.
    dt_sim : simulation timestep.
    seed : RNG seed.
    alpha : skewness parameter for the quadratic noise deformation.
            alpha=0 recovers demo1 exactly.

    Returns
    -------
    r_arr, t_arr : measurement grids.
    xi_all : shape (3, n_t, n_r) — two-point correlators for PAIRS.
    var0_all : shape (3, n_t)    — variance at origin.
    mu2_all : shape (2, n_t)     — <eta_tilde_a(0, t)^2> per component.
    mu3_all : shape (2, n_t)     — <eta_tilde_a(0, t)^3> per component.
    n_good  : number of non-blown realisations.
    """
    rng = np.random.default_rng(seed)
    dx = sigma_x / 5.0
    max_r = max(r_values) if len(r_values) else 0
    n_sites = int(np.ceil(max_r / dx)) + 1
    x_grid = np.arange(n_sites) * dx
    t_max_s = max(t_measure_list) + 1.0
    n_steps = int(t_max_s / dt_sim)

    # Spatial Cholesky (small matrix, kept)
    K_space = np.exp(-np.abs(x_grid[:, None] - x_grid[None, :]) / sigma_x)
    L_space = np.linalg.cholesky(K_space)

    # AR(1) parameters for temporal correlation
    rho = np.exp(-dt_sim / sigma_t)
    sig_innov = np.sqrt(lam * (1 - rho**2))
    sig_init = np.sqrt(lam)

    t_meas_idx = {t: int(round(t / dt_sim)) for t in t_measure_list}

    print(f"  Grid: {n_sites} sites, {n_steps} steps (dt={dt_sim})")
    print(f"  AR(1): rho={rho:.6f}, sigma_innov={sig_innov:.6f}")
    print(f"  Non-Gaussian deformation: alpha={alpha}")

    # Accumulators for xi, var0
    cross = {(a, b): {t: np.zeros(n_sites) for t in t_measure_list}
             for a, b in PAIRS}
    var0 = {(a, b): {t: 0.0 for t in t_measure_list}
            for a, b in PAIRS}
    # Accumulators for noise moments at x=0
    mu2 = {a: {t: 0.0 for t in t_measure_list} for a in range(2)}
    mu3 = {a: {t: 0.0 for t in t_measure_list} for a in range(2)}

    batch_size = 500
    n_batches = n_real // batch_size

    n_good = 0
    t0 = time.time()
    for bi in range(n_batches):
        # Initial Gaussian noise at t=0
        z1_prev = sig_init * (L_space @ rng.standard_normal((n_sites, batch_size)))
        z2_prev = sig_init * (L_space @ rng.standard_normal((n_sites, batch_size)))
        # Apply quadratic deformation to obtain eta_tilde used in the Langevin step.
        eta1_k = _deform(z1_prev.T, alpha, lam)
        eta2_k = _deform(z2_prev.T, alpha, lam)

        phi = np.zeros((2, batch_size, n_sites))
        blown = np.zeros(batch_size, dtype=bool)

        for k in range(n_steps):
            # Advance Gaussian noise via AR(1)
            z1_next = rho * z1_prev + sig_innov * (
                L_space @ rng.standard_normal((n_sites, batch_size)))
            z2_next = rho * z2_prev + sig_innov * (
                L_space @ rng.standard_normal((n_sites, batch_size)))
            eta1_kn = _deform(z1_next.T, alpha, lam)
            eta2_kn = _deform(z2_next.T, alpha, lam)

            # Heun step (same drift as demo1)
            f1 = -phi[0] + phi[1]**2 + eta1_k
            f2 = -phi[1] + phi[0] * phi[1] + eta2_k
            phi_pred = phi.copy()
            phi_pred[0] += dt_sim * f1
            phi_pred[1] += dt_sim * f2
            g1 = -phi_pred[0] + phi_pred[1]**2 + eta1_kn
            g2 = -phi_pred[1] + phi_pred[0] * phi_pred[1] + eta2_kn
            phi[0] += dt_sim / 2 * (f1 + g1)
            phi[1] += dt_sim / 2 * (f2 + g2)

            new_blown = np.any(np.abs(phi[0]) > 1e6, axis=1) | \
                        np.any(np.abs(phi[1]) > 1e6, axis=1)
            blown |= new_blown

            # Accumulate at measurement times
            t_k = (k + 1) * dt_sim
            for t_m in t_measure_list:
                if abs(t_k - t_m) < dt_sim / 2:
                    good = ~blown
                    # xi and var0 accumulators
                    for a, b in PAIRS:
                        cross[(a, b)][t_m] += np.sum(
                            phi[a, good, 0:1] * phi[b, good], axis=0
                        )
                        var0[(a, b)][t_m] += np.sum(
                            phi[a, good, 0] * phi[b, good, 0]
                        )
                    # Noise moments at x=0 (use the *next-step* noise, which is
                    # what drives phi at this measurement time — consistent with
                    # the Heun predictor evaluation above).
                    for a in range(2):
                        eta_a0 = (eta1_kn if a == 0 else eta2_kn)[good, 0]
                        mu2[a][t_m] += np.sum(eta_a0**2)
                        mu3[a][t_m] += np.sum(eta_a0**3)

            # Shift noise
            eta1_k = eta1_kn
            eta2_k = eta2_kn
            z1_prev = z1_next
            z2_prev = z2_next

        n_good += int(np.sum(~blown))
        if (bi + 1) % 10 == 0:
            elapsed = time.time() - t0
            n_blown_total = (bi + 1) * batch_size - n_good
            print(f"  batch {bi+1}/{n_batches}  ({elapsed:.0f}s)"
                  f"  [{n_blown_total} blown]")

    print(f"  Used {n_good}/{n_real} realisations"
          f" ({n_real - n_good} diverged)")

    # Pack results
    r_arr = np.array(r_values)
    t_arr = np.array(t_measure_list)
    xi_all = np.zeros((len(PAIRS), len(t_measure_list), len(r_values)))
    var0_all = np.zeros((len(PAIRS), len(t_measure_list)))
    mu2_all = np.zeros((2, len(t_measure_list)))
    mu3_all = np.zeros((2, len(t_measure_list)))

    for ip, (a, b) in enumerate(PAIRS):
        for it, t_m in enumerate(t_measure_list):
            G_sites = cross[(a, b)][t_m] / n_good
            xi_all[ip, it, :] = np.interp(r_values, x_grid, G_sites)
            var0_all[ip, it] = var0[(a, b)][t_m] / n_good
    for a in range(2):
        for it, t_m in enumerate(t_measure_list):
            mu2_all[a, it] = mu2[a][t_m] / n_good
            mu3_all[a, it] = mu3[a][t_m] / n_good

    return r_arr, t_arr, xi_all, var0_all, mu2_all, mu3_all, n_good


def _deform(eta: np.ndarray, alpha: float, var_eta: float) -> np.ndarray:
    """Apply quadratic non-Gaussian deformation elementwise.

    eta_tilde = eta + alpha * (eta**2 - var_eta)

    Subtracting var_eta keeps eta_tilde zero-mean. alpha=0 is the identity.
    """
    if alpha == 0.0:
        return eta
    return eta + alpha * (eta * eta - var_eta)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Langevin simulation with non-Gaussian noise (demo2).")
    parser.add_argument("--n_real", type=int, default=50_000)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="skewness strength (alpha=0 recovers demo1)")
    parser.add_argument("-o", "--output", default="sim_cache.npz")
    args = parser.parse_args()

    # Build combined evaluation grid (covers all three figures)
    r_vals_fig1 = [0.0, 0.4, 1.0]
    t_vals_fig2 = [1.0, 5.0, 15.0]
    r_grid_fig2 = np.linspace(0.0, 2.5, 11).tolist()
    t_grid_fig1 = np.logspace(np.log10(0.1), np.log10(T_MAX), 15).tolist()

    all_t = sorted(set(t_grid_fig1 + t_vals_fig2))
    all_r = sorted(set(r_grid_fig2 + r_vals_fig1))

    print(f"Grid: {len(all_r)} r-values x {len(all_t)} t-values")
    print(f"Realisations: {args.n_real}, dt={args.dt}, "
          f"seed={args.seed}, alpha={args.alpha}")
    print("Running simulation...")

    t_start = time.time()
    r_arr, t_arr, xi_all, var0_all, mu2_all, mu3_all, n_good = \
        simulate_correlators(
            all_r, all_t,
            n_real=args.n_real, dt_sim=args.dt,
            seed=args.seed, alpha=args.alpha,
        )
    elapsed = time.time() - t_start
    print(f"Done in {elapsed:.1f}s")

    np.savez(
        args.output,
        r=r_arr,
        t=t_arr,
        xi=xi_all,
        var0=var0_all,
        mu2=mu2_all,
        mu3=mu3_all,
        pairs=np.array(PAIRS),
        n_real=n_good,
        dt_sim=args.dt,
        seed=args.seed,
        alpha=args.alpha,
        lam=lam,
        sigma_t=sigma_t,
        sigma_x=sigma_x,
    )
    print(f"Saved to {args.output} ({xi_all.nbytes / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
