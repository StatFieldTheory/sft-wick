"""Reproduce the figures from ``examples/demo1/analysis.ipynb`` using
the L2 (YAML) workflow API.

The notebook's perturbative xi_ab(r, t) values come from a
hand-built loop over ``compute_moment`` + a custom Gauss-Legendre
integrator. Here we drive the same physics through
``sft-wick run config.yaml`` (programmatically): the L2 sweep
returns a ``totals`` DataFrame keyed by ``(x, y, t_final, a, b,
order)``, which we slice to rebuild the same four figures with the
same plot setup as the notebook.

Output PDFs land in ``L2/figures/`` and should be visually
indistinguishable from ``../figures/{waterfall,xi_vs_time,
comparison_multi_time,xi_vs_order}.pdf``.

The Langevin simulation in the notebook is independent of the
sft-wick API. We re-use the cached realisations from
``../sim_cache.npz`` (produced by the notebook's batch run) and
re-run the single-realisation snapshot for the waterfall plot
locally with the same RNG seed.

Run::

    cd examples/demo1/L2
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        python reproduce_figures.py

The thread caps are important because ``config.yaml`` sets
``sweep.n_jobs: -1`` (one worker per CPU core); without the caps a
multi-threaded BLAS would multiply by the core count and
oversubscribe the machine -- often resulting in a slower run than
``n_jobs: 1``.
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sft_wick.workflow.config import load_workflow_config, run_workflow

THIS_DIR = Path(__file__).resolve().parent
DEMO_DIR = THIS_DIR.parent
CONFIG = THIS_DIR / "config.yaml"
# Local sim cache (T_MAX=100, N=100K) -- matches the notebook's
# in-memory parameters. The package-level ``demo1/sim_cache.npz``
# is from an older 2K-realisation run with a different t-grid;
# don't touch that.
SIM_CACHE = THIS_DIR / "sim_cache_l2.npz"
FIG_DIR = THIS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


# Simulation parameters (verbatim from analysis.ipynb).
SIM_N_REAL = 100_000
SIM_DT = 0.02
SIM_SEED = 12


# ---------- Physical parameters (must match config.yaml + notebook) ----
LAM = 0.05
SIGMA_T = 0.3
SIGMA_X = 1.0
GAMMA = 1.0
N_COMP = 2
T_MAX = 100.0


# ---------- Notebook's global font sizes (verbatim) -------------------
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "legend.title_fontsize": 11,
    "figure.titlesize": 15,
})


# ---------- Plot style constants (verbatim from notebook) -------------
PAIRS = [(0, 0), (1, 1)]
ORDERS = [0, 2, 4]
ORDER_COLORS = {0: "#4e79a7", 2: "#59a14f", 4: "#e15759"}
ORDER_LABELS = {
    0: r"$\xi^{(0)}$",
    2: r"$\xi^{(0{+}2)}$",
    4: r"$\xi^{(0{+}2{+}4)}$",
}
PAIR_COLORS = {(0, 0): "#1b9e77", (1, 1): "#d95f02"}
PAIR_MARKERS = {(0, 0): "o", (1, 1): "s"}
PAIR_LABELS = {(0, 0): r"$\xi_{11}$", (1, 1): r"$\xi_{22}$"}


# =====================================================================
# Step 1: run the L2 workflow and unpack totals into a (a, b, r, t,
# order) lookup compatible with the notebook's ``pert_by_t`` /
# ``pert_fig2`` / ``order_vals`` structures.
# =====================================================================


def run_l2_sweep():
    """Load + execute config.yaml. Returns the SweepResult totals
    DataFrame indexed by (x, y, t_final, a, b, order)."""
    print(f"[L2] loading config: {CONFIG}")
    cfg = load_workflow_config(CONFIG)
    t0 = time.perf_counter()
    sweep, totals = run_workflow(cfg)
    print(f"[L2] sweep done in {time.perf_counter() - t0:.1f}s "
          f"({len(totals)} totals rows)")
    return totals


def lookup(totals, a: int, b: int, r: float, t: float, order: int) -> float:
    """Pick the totals row at the requested (a, b, y=r, t_final=t,
    order). Tolerant float match -- snaps to the closest available
    grid value (rtol=1e-9 abs+rel) so we don't trip on harmless
    float drift between the YAML grid and the figure code."""
    df = totals
    base = (df["a"] == a) & (df["b"] == b) & (df["order"] == order)
    mask = base & np.isclose(df["y"].to_numpy(), r, atol=1e-9, rtol=1e-9) \
                & np.isclose(df["t_final"].to_numpy(), t, atol=1e-9, rtol=1e-9)
    sub = df.loc[mask, "value"]
    if len(sub) != 1:
        # Diagnostic: surface the closest grid points so the error
        # message is actionable when the YAML and figure grids drift.
        sub_pair = df.loc[base]
        ys = np.unique(sub_pair["y"].to_numpy())
        ts = np.unique(sub_pair["t_final"].to_numpy())
        ny = ys[np.argmin(np.abs(ys - r))]
        nt = ts[np.argmin(np.abs(ts - t))]
        raise KeyError(
            f"no unique row for a={a}, b={b}, r={r}, t={t}, "
            f"order={order} (got {len(sub)} matches). "
            f"Closest grid: y={ny}, t={nt}."
        )
    return float(sub.iloc[0])


# =====================================================================
# Step 2: reload the simulation cache (produced by the notebook).
# =====================================================================


def simulate_correlators(r_values, t_measure_list,
                         n_real=SIM_N_REAL, dt_sim=SIM_DT, seed=SIM_SEED):
    """Heun (RK2) Langevin simulation with AR(1) noise generation.

    Verbatim from ``analysis.ipynb`` -- produces xi_ab(r, t) for
    pairs (00, 01, 11) on the requested (r, t) grid by averaging
    ``n_real`` realisations.
    """
    F = np.zeros((N_COMP, N_COMP, N_COMP))
    F[0, 1, 1] = 1.0
    F[1, 0, 1] = 0.5
    F[1, 1, 0] = 0.5

    rng = np.random.default_rng(seed)
    dx = SIGMA_X / 5.0
    max_r = max(r_values) if r_values else 0
    n_sites = int(np.ceil(max_r / dx)) + 1
    x_grid = np.arange(n_sites) * dx
    t_max_s = max(t_measure_list) + 1.0
    n_steps = int(t_max_s / dt_sim)

    L_space = np.linalg.cholesky(
        np.exp(-np.abs(x_grid[:, None] - x_grid[None, :]) / SIGMA_X))
    rho = np.exp(-dt_sim / SIGMA_T)
    sig_innov = np.sqrt(LAM * (1 - rho ** 2))
    sig_init = np.sqrt(LAM)

    PAIRS_SIM = [(0, 0), (0, 1), (1, 1)]
    cross = {p: {t: np.zeros(n_sites) for t in t_measure_list} for p in PAIRS_SIM}
    var0 = {p: {t: 0.0 for t in t_measure_list} for p in PAIRS_SIM}

    batch_size = 500
    n_batches = n_real // batch_size
    n_good = 0
    t0 = time.time()

    for bi in range(n_batches):
        z1_prev = sig_init * (L_space @ rng.standard_normal((n_sites, batch_size)))
        z2_prev = sig_init * (L_space @ rng.standard_normal((n_sites, batch_size)))
        eta1_k, eta2_k = z1_prev.T, z2_prev.T

        phi = np.zeros((2, batch_size, n_sites))
        blown = np.zeros(batch_size, dtype=bool)

        for k in range(n_steps):
            z1_next = rho * z1_prev + sig_innov * (L_space @ rng.standard_normal((n_sites, batch_size)))
            z2_next = rho * z2_prev + sig_innov * (L_space @ rng.standard_normal((n_sites, batch_size)))
            eta1_kn, eta2_kn = z1_next.T, z2_next.T

            eta_k = np.stack([eta1_k, eta2_k])
            eta_kn = np.stack([eta1_kn, eta2_kn])
            drift = (-GAMMA * phi
                     + np.einsum("abc,bji,cji->aji", F, phi, phi)
                     + eta_k)
            phi_p = phi + dt_sim * drift
            drift_p = (-GAMMA * phi_p
                       + np.einsum("abc,bji,cji->aji", F, phi_p, phi_p)
                       + eta_kn)
            phi += dt_sim / 2 * (drift + drift_p)

            blown |= np.any(np.abs(phi).max(axis=0) > 1e6, axis=1)

            t_k = (k + 1) * dt_sim
            for t_m in t_measure_list:
                if abs(t_k - t_m) < dt_sim / 2:
                    good = ~blown
                    for a, b in PAIRS_SIM:
                        cross[(a, b)][t_m] += np.sum(
                            phi[a, good, 0:1] * phi[b, good], axis=0)
                        var0[(a, b)][t_m] += np.sum(
                            phi[a, good, 0] * phi[b, good, 0])

            eta1_k, eta2_k = eta1_kn, eta2_kn
            z1_prev, z2_prev = z1_next, z2_next

        n_good += int(np.sum(~blown))
        if (bi + 1) % 20 == 0:
            print(f"  batch {bi+1}/{n_batches}  ({time.time() - t0:.0f}s)"
                  f"  [{(bi+1)*batch_size - n_good} blown]")

    r_arr = np.array(r_values)
    t_arr = np.array(t_measure_list)
    xi_all = np.zeros((len(PAIRS_SIM), len(t_measure_list), len(r_values)))
    var0_all = np.zeros((len(PAIRS_SIM), len(t_measure_list)))
    for ip, (a, b) in enumerate(PAIRS_SIM):
        for it, t_m in enumerate(t_measure_list):
            xi_all[ip, it, :] = np.interp(
                r_values, x_grid, cross[(a, b)][t_m] / n_good,
            )
            var0_all[ip, it] = var0[(a, b)][t_m] / n_good
    return r_arr, t_arr, xi_all, var0_all, n_good


def load_sim_cache():
    """Load (or generate + cache) the L2 simulation. Returns
    ``(sim_t, sim_data, sim_n_real, sim_r)``.

    First run takes ~4 min (100K realisations). Subsequent runs
    load instantly from ``sim_cache_l2.npz``.
    """
    if not SIM_CACHE.exists():
        print(f"[sim] no cache found at {SIM_CACHE}; generating "
              f"({SIM_N_REAL:,} realisations, dt={SIM_DT})...")
        # Same (r, t) grid the notebook builds in-memory.
        _r_grid = np.linspace(0.0, 2.5, 11).tolist() + [0.4, 0.5, 1.0]
        _t_grid = (np.logspace(np.log10(0.1), np.log10(T_MAX), 20).tolist()
                   + [1.0, 5.0, 15.0])
        all_r = sorted(set(_r_grid))
        all_t = sorted(set(_t_grid))
        sim_r, sim_t, sim_xi, sim_var0, sim_n_real = simulate_correlators(
            all_r, all_t,
        )
        np.savez(
            SIM_CACHE,
            r=sim_r, t=sim_t, xi=sim_xi, var0=sim_var0,
            pairs=np.array([(0, 0), (0, 1), (1, 1)]),
            n_real=sim_n_real, dt_sim=SIM_DT, seed=SIM_SEED,
        )
        print(f"[sim] saved to {SIM_CACHE}")

    raw = np.load(SIM_CACHE)
    sim_r = raw["r"]
    sim_t = raw["t"]
    sim_xi = raw["xi"]
    sim_var0 = raw["var0"]
    sim_n_real = int(raw["n_real"])
    pair_idx = {(0, 0): 0, (0, 1): 1, (1, 1): 2}
    sim_data = {
        pair: {"xi": sim_xi[idx], "var0": sim_var0[idx]}
        for pair, idx in pair_idx.items()
    }
    return sim_t, sim_data, sim_n_real, sim_r


# =====================================================================
# Step 3: Single-realisation Langevin simulation for the waterfall.
# Verbatim copy from analysis.ipynb -- no sft-wick API involved.
# =====================================================================


def single_realisation(n_sites=60, dt_sim=0.01, t_max=T_MAX, seed=0):
    F = np.zeros((N_COMP, N_COMP, N_COMP))
    F[0, 1, 1] = 1.0
    F[1, 0, 1] = 0.5
    F[1, 1, 0] = 0.5

    rng = np.random.default_rng(seed)
    dx = SIGMA_X / 8.0
    x_grid = np.arange(n_sites) * dx
    n_steps = int(t_max / dt_sim)

    L_space = np.linalg.cholesky(
        np.exp(-np.abs(x_grid[:, None] - x_grid[None, :]) / SIGMA_X))
    rho = np.exp(-dt_sim / SIGMA_T)
    sig_innov = np.sqrt(LAM * (1 - rho ** 2))
    sig_init = np.sqrt(LAM)

    save_every = max(1, int(0.2 / dt_sim))
    phi1_list, phi2_list, eta1_list, eta2_list, t_list = [], [], [], [], []

    z1_prev = sig_init * (L_space @ rng.standard_normal(n_sites))
    z2_prev = sig_init * (L_space @ rng.standard_normal(n_sites))
    eta1_k, eta2_k = z1_prev.copy(), z2_prev.copy()
    phi1, phi2 = np.zeros(n_sites), np.zeros(n_sites)

    phi1_list.append(phi1.copy())
    phi2_list.append(phi2.copy())
    eta1_list.append(eta1_k.copy())
    eta2_list.append(eta2_k.copy())
    t_list.append(0.0)

    for k in range(n_steps):
        z1_next = rho * z1_prev + sig_innov * (L_space @ rng.standard_normal(n_sites))
        z2_next = rho * z2_prev + sig_innov * (L_space @ rng.standard_normal(n_sites))
        eta1_kn, eta2_kn = z1_next, z2_next

        phi_vec = np.stack([phi1, phi2])
        eta_vec = np.stack([eta1_k, eta2_k])
        eta_vec_n = np.stack([eta1_kn, eta2_kn])
        drift = (-GAMMA * phi_vec
                 + np.einsum("abc,b...,c...->a...", F, phi_vec, phi_vec)
                 + eta_vec)
        phi_p = phi_vec + dt_sim * drift
        drift_p = (-GAMMA * phi_p
                   + np.einsum("abc,b...,c...->a...", F, phi_p, phi_p)
                   + eta_vec_n)
        phi_vec = phi_vec + dt_sim / 2 * (drift + drift_p)
        phi1, phi2 = phi_vec[0], phi_vec[1]

        eta1_k, eta2_k = eta1_kn, eta2_kn
        z1_prev, z2_prev = z1_next, z2_next

        if (k + 1) % save_every == 0:
            phi1_list.append(phi1.copy())
            phi2_list.append(phi2.copy())
            eta1_list.append(eta1_k.copy())
            eta2_list.append(eta2_k.copy())
            t_list.append((k + 1) * dt_sim)

    return (
        x_grid, np.array(t_list),
        np.array(phi1_list), np.array(phi2_list),
        np.array(eta1_list), np.array(eta2_list),
    )


# =====================================================================
# Figure 0: waterfall.pdf
# =====================================================================


def figure_waterfall():
    print("[fig] waterfall.pdf")
    x_wf, t_wf, phi1_wf, phi2_wf, eta1_wf, eta2_wf = single_realisation()

    fig = plt.figure(figsize=(14, 11))
    gs = GridSpec(
        4, 2, figure=fig,
        width_ratios=[40, 1],
        hspace=0.18, wspace=0.04,
    )
    ax_phi1 = fig.add_subplot(gs[0, 0])
    ax_phi2 = fig.add_subplot(gs[1, 0], sharex=ax_phi1, sharey=ax_phi1)
    ax_eta1 = fig.add_subplot(gs[2, 0], sharex=ax_phi1, sharey=ax_phi1)
    ax_eta2 = fig.add_subplot(gs[3, 0], sharex=ax_phi1, sharey=ax_phi1)
    cax_phi = fig.add_subplot(gs[0:2, 1])
    cax_eta = fig.add_subplot(gs[2:4, 1])

    extent = [t_wf[0], t_wf[-1], x_wf[0], x_wf[-1]]
    vmax_phi = max(np.abs(phi1_wf).max(), np.abs(phi2_wf).max())
    vmax_eta = max(np.abs(eta1_wf).max(), np.abs(eta2_wf).max())
    kw_phi = dict(aspect="auto", origin="lower", cmap="RdBu_r",
                  vmin=-vmax_phi, vmax=vmax_phi, extent=extent)
    kw_eta = dict(aspect="auto", origin="lower", cmap="PuOr_r",
                  vmin=-vmax_eta, vmax=vmax_eta, extent=extent)

    ax_phi1.imshow(phi1_wf.T, **kw_phi)
    im_phi = ax_phi2.imshow(phi2_wf.T, **kw_phi)
    ax_eta1.imshow(eta1_wf.T, **kw_eta)
    im_eta = ax_eta2.imshow(eta2_wf.T, **kw_eta)

    for ax in (ax_phi1, ax_phi2, ax_eta1, ax_eta2):
        ax.set_ylabel(r"$x / \sigma_x$", fontsize=14)
    for ax in (ax_phi1, ax_phi2, ax_eta1):
        ax.tick_params(labelbottom=False)
    ax_eta2.set_xlabel("$t$", fontsize=14)

    label_kw = dict(fontsize=14, va="top", fontweight="bold",
                    bbox=dict(fc="white", alpha=0.75, ec="none"))
    ax_phi1.text(0.01, 0.92, r"$\varphi_1(x, t)$", transform=ax_phi1.transAxes, **label_kw)
    ax_phi2.text(0.01, 0.92, r"$\varphi_2(x, t)$", transform=ax_phi2.transAxes, **label_kw)
    ax_eta1.text(0.01, 0.92, r"$\eta_1(x, t)$", transform=ax_eta1.transAxes, **label_kw)
    ax_eta2.text(0.01, 0.92, r"$\eta_2(x, t)$", transform=ax_eta2.transAxes, **label_kw)

    cb_phi = fig.colorbar(im_phi, cax=cax_phi)
    cb_phi.set_label(r"$\varphi$ amplitude", fontsize=13)
    cb_eta = fig.colorbar(im_eta, cax=cax_eta)
    cb_eta.set_label(r"$\eta$ amplitude", fontsize=13)

    fig.savefig(FIG_DIR / "waterfall.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)


# =====================================================================
# Figure 1: xi_vs_time.pdf
# =====================================================================


def figure_xi_vs_time(totals, sim_t, sim_data, sim_n_real, sim_r):
    print("[fig] xi_vs_time.pdf")
    r_fixed = 0.5
    t_grid = np.array(sorted(set(totals["t_final"].tolist())))
    # Re-derive the 25-point log grid the notebook used; the union
    # of the YAML t list also includes 1.0 / 3.0 / 15.0 / 30.0 from
    # the other figures, but those happen to lie on the log grid.
    t_grid = np.logspace(np.log10(0.1), np.log10(T_MAX), 25)

    pert_by_t = {pair: {o: np.zeros(len(t_grid)) for o in ORDERS}
                 for pair in PAIRS}
    for ti, t_f in enumerate(t_grid):
        for a, b in PAIRS:
            for o in ORDERS:
                pert_by_t[(a, b)][o][ti] = lookup(totals, a, b, r_fixed, t_f, o)

    xi_pert_cumul = {
        (a, b): sum(pert_by_t[(a, b)][o] for o in ORDERS) for a, b in PAIRS
    }
    ri_sim = int(np.argmin(np.abs(sim_r - r_fixed)))
    xi_sim = {
        (a, b): np.interp(t_grid, sim_t, sim_data[(a, b)]["xi"][:, ri_sim])
        for a, b in PAIRS
    }

    # Error envelope (same as notebook): MC noise + Heun bias.
    dt_sim = 0.02
    mc_noise = {}
    for a, b in PAIRS:
        xi0 = np.interp(t_grid, sim_t, sim_data[(a, b)]["xi"][:, 0])
        mc_noise[(a, b)] = np.sqrt(xi0 ** 2 + xi_sim[(a, b)] ** 2) / np.sqrt(sim_n_real)
    heun_bias = {(a, b): xi_pert_cumul[(a, b)] * GAMMA * dt_sim ** 2 for a, b in PAIRS}
    total_noise = {
        (a, b): np.sqrt(mc_noise[(a, b)] ** 2 + heun_bias[(a, b)] ** 2)
        for a, b in PAIRS
    }

    fig, axes = plt.subplots(
        2, 2, figsize=(11, 6.5),
        gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.06, "wspace": 0.08},
        sharex=True, sharey="row",
    )

    for col, (a, b) in enumerate(PAIRS):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        cumul = np.zeros(len(t_grid))
        order_data = []
        for order in ORDERS:
            cumul = cumul + pert_by_t[(a, b)][order]
            order_data.append((cumul.copy(), order))

        for cumul_arr, order in reversed(order_data):
            if order == max(ORDERS):
                lw, ls, alpha = 3.0, "-", 0.7
            elif order == 2:
                lw, ls, alpha = 2.0, "--", 0.8
            else:
                lw, ls, alpha = 2.2, ":", 0.9
            ax_top.plot(
                t_grid, cumul_arr, color=ORDER_COLORS[order],
                lw=lw, ls=ls, alpha=alpha,
                label=ORDER_LABELS[order] if col == 0 else None,
            )

        t_mask = (sim_t >= t_grid[0]) & (sim_t <= t_grid[-1])
        t_pts = sim_t[t_mask]
        v_pts = sim_data[(a, b)]["xi"][t_mask, ri_sim]
        e_pts = np.sqrt(
            sim_data[(a, b)]["xi"][t_mask, 0] ** 2 + v_pts ** 2
        ) / np.sqrt(sim_n_real)
        ax_top.errorbar(
            t_pts, v_pts, yerr=e_pts,
            fmt="o", color="k", ms=3.5, lw=0.7,
            capsize=1.5, capthick=0.5, zorder=5,
            label="Simulation" if col == 0 else None,
        )

        ax_top.set_xscale("log")
        ax_top.tick_params(labelbottom=False)
        ax_top.text(
            0.03, 0.93, PAIR_LABELS[(a, b)],
            transform=ax_top.transAxes, fontsize=15,
            va="top", fontweight="bold",
        )
        if col == 0:
            ax_top.set_ylabel(r"$\xi_{ab}(r,\,t)$")

        residual = xi_pert_cumul[(a, b)] - xi_sim[(a, b)]
        ax_bot.plot(
            t_grid, residual, "k-", lw=1.1,
            label=r"$\Delta\xi$" if col == 0 else None,
        )
        env = total_noise[(a, b)]
        ax_bot.fill_between(
            t_grid, -env, env,
            color="#bbbbbb", alpha=0.5,
            label=r"MC $\oplus$ Heun" if col == 0 else None,
        )
        ax_bot.axhline(0, color="gray", lw=0.4)
        ax_bot.set_xscale("log")
        ax_bot.set_xlabel("$t$")
        if col == 0:
            ax_bot.set_ylabel(r"$\xi_{\rm pert} - \xi_{\rm sim}$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[1, 0].get_legend_handles_labels()
    handles += h2
    labels += l2
    fig.legend(
        handles, labels, loc="upper center",
        ncol=len(handles), fontsize=13, framealpha=0.9,
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "xi_vs_time.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)


# =====================================================================
# Figure 2: comparison_multi_time.pdf
# =====================================================================


def figure_comparison_multi_time(totals, sim_t, sim_data, sim_r):
    print("[fig] comparison_multi_time.pdf")
    t_vals_fig2 = [1.0, 15.0, 30.0]
    r_grid_fig2 = np.linspace(0.0, 2.5, 11)

    pert_fig2 = {pair: {} for pair in PAIRS}
    for t_f in t_vals_fig2:
        for a, b in PAIRS:
            order_data = {o: np.zeros(len(r_grid_fig2)) for o in ORDERS}
            for ri, r in enumerate(r_grid_fig2):
                for o in ORDERS:
                    order_data[o][ri] = lookup(totals, a, b, r, t_f, o)
            pert_fig2[(a, b)][t_f] = order_data

    fig, axes = plt.subplots(
        len(PAIRS), len(t_vals_fig2),
        figsize=(4.0 * len(t_vals_fig2), 3.2 * len(PAIRS)),
        sharex=True, sharey="row",
    )
    for col, t_f in enumerate(t_vals_fig2):
        ti = int(np.argmin(np.abs(sim_t - t_f)))
        for row, (a, b) in enumerate(PAIRS):
            ax = axes[row, col]
            od = pert_fig2[(a, b)][t_f]

            cumul = od[0] + od[2] + od[4]
            ax.plot(
                r_grid_fig2, cumul, color=ORDER_COLORS[4],
                lw=3.0, alpha=0.7,
                label=ORDER_LABELS[4] if col == 0 and row == 0 else None,
            )
            ax.plot(
                r_grid_fig2, od[0], color=ORDER_COLORS[0],
                lw=2.2, ls=":", alpha=0.9,
                label=ORDER_LABELS[0] if col == 0 and row == 0 else None,
            )

            sim_vals = np.interp(
                r_grid_fig2, sim_r, sim_data[(a, b)]["xi"][ti, :],
            )
            mk = PAIR_MARKERS[(a, b)]
            ax.plot(
                r_grid_fig2, sim_vals, mk, color="k", ms=5,
                mfc="none", mew=1.2, zorder=5,
                label="Simulation" if col == 0 and row == 0 else None,
            )

            if col == 0:
                ax.set_ylabel(PAIR_LABELS[(a, b)])
            if row == 0:
                ax.text(
                    0.97, 0.93, f"$t = {t_f:.0f}$",
                    transform=ax.transAxes, fontsize=14,
                    va="top", ha="right", fontweight="bold",
                )
            if row == len(PAIRS) - 1:
                ax.set_xlabel(r"$r / \sigma_x$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center",
        ncol=len(handles), fontsize=13, framealpha=0.9,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "comparison_multi_time.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)


# =====================================================================
# Figure 3: xi_vs_order.pdf
# =====================================================================


def figure_xi_vs_order(totals, sim_t, sim_data, sim_r):
    print("[fig] xi_vs_order.pdf")
    r_pt, t_pt = 0.4, 3.0

    order_vals = {
        (a, b): {o: lookup(totals, a, b, r_pt, t_pt, o) for o in ORDERS}
        for a, b in PAIRS
    }

    def sim_at(a, b, r, t):
        ri = int(np.argmin(np.abs(sim_r - r)))
        ti = int(np.argmin(np.abs(sim_t - t)))
        return sim_data[(a, b)]["xi"][ti, ri]

    sim_pt = {pair: sim_at(*pair, r_pt, t_pt) for pair in PAIRS}

    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.5), gridspec_kw={"wspace": 0.3},
    )
    x = np.arange(len(ORDERS))
    bw = 0.35
    pair_hatches = {(0, 0): "///", (1, 1): ""}

    ax = axes[0]
    for ip, (a, b) in enumerate(PAIRS):
        vals = order_vals[(a, b)]
        cumul = np.cumsum([vals[o] for o in ORDERS])
        ax.bar(
            x + ip * bw, cumul, bw,
            color=PAIR_COLORS[(a, b)], edgecolor="k", lw=0.6,
            hatch=pair_hatches[(a, b)], alpha=0.85,
            label=PAIR_LABELS[(a, b)],
        )
        ax.axhline(
            sim_pt[(a, b)], color=PAIR_COLORS[(a, b)],
            ls="--", lw=2, alpha=0.6,
            label=PAIR_LABELS[(a, b)] + " sim",
        )
    ax.set_xticks(x + bw / 2)
    ax.set_xticklabels([ORDER_LABELS[o] for o in ORDERS])
    ax.set_ylabel(rf"$\xi_{{ab}}(r{{=}}{r_pt},\, t{{=}}{t_pt:.0f})$")
    h, l = ax.get_legend_handles_labels()
    by_label = dict(zip(l, h))
    ax.legend(by_label.values(), by_label.keys(), loc="lower right", fontsize=13)
    ax.text(
        0.03, 1.05, "(a) Cumulative sum",
        transform=ax.transAxes, va="top", fontweight="bold",
    )

    ax = axes[1]
    for ip, (a, b) in enumerate(PAIRS):
        vals = order_vals[(a, b)]
        contribs = [abs(vals[o]) for o in ORDERS]
        ax.bar(
            x + ip * bw, contribs, bw,
            color=PAIR_COLORS[(a, b)], edgecolor="k", lw=0.6,
            hatch=pair_hatches[(a, b)], alpha=0.85,
            label=PAIR_LABELS[(a, b)],
        )
    ax.set_yscale("log")
    ax.set_xticks(x + bw / 2)
    ax.set_xticklabels([f"$n={o}$" for o in ORDERS])
    ax.set_ylabel(r"$|\xi^{(n)}_{ab}|$")
    ax.legend(loc="upper right", fontsize=13)
    ax.text(
        0.03, 1.05, "(b) Per-order contribution",
        transform=ax.transAxes, va="top", fontweight="bold",
    )

    fig.savefig(FIG_DIR / "xi_vs_order.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)


# =====================================================================
# Driver
# =====================================================================


def _check_blas_thread_caps() -> None:
    """Print a one-line nudge when ``sweep.n_jobs > 1`` is being
    used without OPENBLAS / MKL / OMP thread caps set."""
    capped = any(
        os.environ.get(name)
        for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
    )
    if not capped:
        print(
            "[reproduce_figures] tip: config.yaml sets sweep.n_jobs=-1; "
            "set OPENBLAS_NUM_THREADS=1 / MKL_NUM_THREADS=1 / "
            "OMP_NUM_THREADS=1 to avoid BLAS oversubscription.",
            file=sys.stderr,
        )


def main():
    _check_blas_thread_caps()
    totals = run_l2_sweep()
    sim_t, sim_data, sim_n_real, sim_r = load_sim_cache()

    figure_waterfall()
    figure_xi_vs_time(totals, sim_t, sim_data, sim_n_real, sim_r)
    figure_comparison_multi_time(totals, sim_t, sim_data, sim_r)
    figure_xi_vs_order(totals, sim_t, sim_data, sim_r)

    print(f"\n[done] figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
