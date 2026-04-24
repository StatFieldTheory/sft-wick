"""Run Langevin simulation and save results for reuse.

Produces: sim_cache.npz containing xi_ab(r, t) for all component pairs
(11, 12, 22) on a combined (r, t) grid covering all three figures.

Uses Heun's method (predictor-corrector / RK2) for O(dt^2) weak
convergence, and an exponential noise kernel exp(-|dt|/tau) generated
via AR(1) recursion (O(n) per realisation, no Cholesky needed).

Usage:
    python run_simulation.py [--n_real 50000] [--dt 0.02] [--seed 123]
"""
import argparse
import numpy as np
import time

# ---------- Physical parameters (must match notebook) ----------
lam      = 0.05
sigma_t  = 0.3
sigma_x  = 1.0
gamma    = 1.0
N_comp   = 2
T_MAX    = 50.0

PAIRS = [(0, 0), (0, 1), (1, 1)]


def _run_batch(batch_size, n_steps, n_sites, dt_sim, L_space,
               rho, sig_innov, sig_init, t_meas_idx, rng_seed):
    """Run one batch of realisations. Designed for joblib parallelism."""
    rng = np.random.default_rng(rng_seed)

    cross = {t: np.zeros(n_sites) for t in t_meas_idx}
    var0 = {t: 0.0 for t in t_meas_idx}

    # Initial noise via AR(1): η(0) ~ N(0, λ) with spatial correlation
    z1_prev = sig_init * (L_space @ rng.standard_normal((n_sites, batch_size)))
    z2_prev = sig_init * (L_space @ rng.standard_normal((n_sites, batch_size)))
    eta1_k = z1_prev.T  # (batch, n_sites)
    eta2_k = z2_prev.T

    phi = np.zeros((2, batch_size, n_sites))
    blown = np.zeros(batch_size, dtype=bool)

    for k in range(n_steps):
        # Next noise step: η(t+dt) = ρ η(t) + σ_innov ε(t)
        z1_next = rho * z1_prev + sig_innov * (L_space @ rng.standard_normal((n_sites, batch_size)))
        z2_next = rho * z2_prev + sig_innov * (L_space @ rng.standard_normal((n_sites, batch_size)))
        eta1_kn = z1_next.T
        eta2_kn = z2_next.T

        # Heun's method
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
        for t_m, t_idx in t_meas_idx.items():
            if k + 1 == t_idx:
                good = ~blown
                for a, b_idx in PAIRS:
                    cross[t_m] = cross.get(t_m, np.zeros(n_sites))
                for a, b_idx in PAIRS:
                    pass  # handled below

        for t_m, t_idx in t_meas_idx.items():
            if k + 1 == t_idx:
                good = ~blown
                break
        else:
            good = None

        if good is not None:
            for t_m, t_idx in t_meas_idx.items():
                if k + 1 == t_idx:
                    cross_ab = {}
                    var0_ab = {}
                    for a, b_idx in PAIRS:
                        cross_ab[(a, b_idx)] = np.sum(
                            phi[a, good, 0:1] * phi[b_idx, good], axis=0
                        )
                        var0_ab[(a, b_idx)] = np.sum(
                            phi[a, good, 0] * phi[b_idx, good, 0]
                        )

        # Shift noise
        eta1_k = eta1_kn
        eta2_k = eta2_kn
        z1_prev = z1_next
        z2_prev = z2_next

    n_good = int(np.sum(~blown))
    return cross, var0, n_good, blown


def simulate_correlators(r_values, t_measure_list,
                         n_real=50_000, dt_sim=0.02, seed=123,
                         n_jobs=1):
    """Heun (RK2) simulation with AR(1) noise generation.

    The exponential kernel exp(-|dt|/sigma_t) is generated as an AR(1)
    process: O(n_steps) per realisation instead of O(n_steps^2).
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

    # Accumulators
    cross = {(a, b): {t: np.zeros(n_sites) for t in t_measure_list}
             for a, b in PAIRS}
    var0  = {(a, b): {t: 0.0 for t in t_measure_list}
             for a, b in PAIRS}

    batch_size = 500
    n_batches = n_real // batch_size

    n_good = 0
    t0 = time.time()
    for bi in range(n_batches):
        # Initial noise
        z1_prev = sig_init * (L_space @ rng.standard_normal((n_sites, batch_size)))
        z2_prev = sig_init * (L_space @ rng.standard_normal((n_sites, batch_size)))
        eta1_k = z1_prev.T
        eta2_k = z2_prev.T

        phi = np.zeros((2, batch_size, n_sites))
        blown = np.zeros(batch_size, dtype=bool)

        for k in range(n_steps):
            # Next noise via AR(1)
            z1_next = rho * z1_prev + sig_innov * (L_space @ rng.standard_normal((n_sites, batch_size)))
            z2_next = rho * z2_prev + sig_innov * (L_space @ rng.standard_normal((n_sites, batch_size)))
            eta1_kn = z1_next.T
            eta2_kn = z2_next.T

            # Heun step
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

            t_k = (k + 1) * dt_sim
            for t_m in t_measure_list:
                if abs(t_k - t_m) < dt_sim / 2:
                    good = ~blown
                    for a, b in PAIRS:
                        cross[(a, b)][t_m] += np.sum(
                            phi[a, good, 0:1] * phi[b, good], axis=0
                        )
                        var0[(a, b)][t_m] += np.sum(
                            phi[a, good, 0] * phi[b, good, 0]
                        )

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

    for ip, (a, b) in enumerate(PAIRS):
        for it, t_m in enumerate(t_measure_list):
            G_sites = cross[(a, b)][t_m] / n_good
            xi_all[ip, it, :] = np.interp(r_values, x_grid, G_sites)
            var0_all[ip, it] = var0[(a, b)][t_m] / n_good

    return r_arr, t_arr, xi_all, var0_all, n_good


def main():
    parser = argparse.ArgumentParser(description="Run Langevin simulation")
    parser.add_argument("--n_real", type=int, default=50_000)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=123)
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
    print(f"Realisations: {args.n_real}, dt={args.dt}, seed={args.seed}")
    print("Running simulation...")

    t_start = time.time()
    r_arr, t_arr, xi_all, var0_all, n_good = simulate_correlators(
        all_r, all_t, n_real=args.n_real, dt_sim=args.dt, seed=args.seed,
    )
    elapsed = time.time() - t_start
    print(f"Done in {elapsed:.1f}s")

    # Save
    np.savez(
        args.output,
        r=r_arr,
        t=t_arr,
        xi=xi_all,           # shape (3, n_t, n_r), pairs ordered as PAIRS
        var0=var0_all,        # shape (3, n_t)
        pairs=np.array(PAIRS),
        n_real=n_good,
        dt_sim=args.dt,
        seed=args.seed,
    )
    print(f"Saved to {args.output} ({xi_all.nbytes / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
