"""Reproduce the figures from ``examples/demo2/analysis.ipynb``
using the L2 (YAML) workflow API.

Demo2's perturbative breakdown has two channels with different
caches:

* **FF** uses ``cache_eff`` -- a Gaussian kernel with renormalised
  variance ``lam_eff = lam * (1 + 2 * alpha^2 * lam)`` (absorbs
  the leading O(alpha^2) variance shift). Driven by
  ``config_FF.yaml``.
* **FK** uses the bare Gaussian (``lam`` not ``lam_eff``) because
  the alpha factor sits in the explicit ``kappa^{(3)}`` of the K
  vertex. Driven by ``config_FK.yaml``.

This script runs both YAMLs, slices the resulting totals
DataFrames by ``vertex_type``, and rebuilds 4 of the 5 figures
saved by analysis.ipynb with the same plot setup. The 5th figure
(``scrutiny_residuals.pdf``) compares alpha=0 vs alpha=0.6 sims
under various truncation levels including order-4 FF and is left
out of the L2 reproduction (it would require an additional
``config_FF_order4.yaml`` and an alpha=0 baseline run -- the
scope is out of proportion with what the L2 demo aims to show).

Integration method
------------------

* **FF** (``config_FF.yaml``): Sobol QMC at ``n_samples=32 768``.
  The order-2 FF integrand is a 2D smooth kernel on a small
  causal sub-simplex; QMC is fast and accurate here.
* **FK** (``config_FK.yaml``): tensor-product Gauss-Legendre at
  ``n_gauss=8`` (4096 deterministic nodes per (t, r, pair) point).
  At large t_f the FK integrand peak is a band of area
  ``~ sigma_t/gamma`` inside a 4-simplex of area ``t_f^2/2``;
  Sobol QMC severely under-resolves the band (factor-of-2 bias
  at t_f=15 even at N=512K), while GL exploits the smooth
  exponential structure for exponential convergence.  The L2 GL
  output matches ``analysis.ipynb``'s hand-derived
  ``_fk_spatial_integral`` to floating-point at matched
  ``n_gauss`` (locked by
  ``tests/test_gauss_legendre_integrator.py``).

Override ``--override sweep.n_gauss=12`` for tighter accuracy at
t_f >> 1; cost grows as ``n_gauss^4`` for FK (12^4 = 20 736 nodes
per point, ~2.5x the default).

Run::

    cd examples/demo2/L2
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
        python reproduce_figures.py

The thread caps are important because each ``config_*.yaml`` sets
``sweep.n_jobs: -1``; without the caps a multi-threaded BLAS
would multiply by the core count and oversubscribe the machine.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sft_wick.workflow.config import load_workflow_config, run_workflow

THIS_DIR = Path(__file__).resolve().parent
DEMO_DIR = THIS_DIR.parent
CONFIG_FF = THIS_DIR / "config_FF.yaml"
CONFIG_FK = THIS_DIR / "config_FK.yaml"
SIM_CACHE = DEMO_DIR / "sim_cache_a0.6.npz"
FIG_DIR = THIS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ---------- Physical parameters (must match the YAMLs + notebook) ----
LAM = 0.05
SIGMA_T = 0.3
SIGMA_X = 1.0
GAMMA = 1.0
N_COMP = 2
ALPHA = 0.6


# ---------- Notebook plot rcParams (verbatim) ------------------------
plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 10, "figure.titlesize": 13,
})


PAIRS = [(0, 0), (0, 1), (1, 1)]
SIM_PAIR_IDX = {(0, 0): 0, (0, 1): 1, (1, 1): 2}


def mc_error(sim: dict, pair: tuple[int, int], series: np.ndarray,
             t_index: int | None = None) -> np.ndarray:
    """Monte-Carlo standard error of a measured ``xi_ab`` series.

    The error on a mean of per-realisation PRODUCTS
    ``phi_a(0) phi_b(r)`` is ``sqrt(Var[phi_a phi_b] / n)``, NOT
    ``sqrt(xi^2 / n)``.  The latter is what this module plotted until
    2026-09; it understates the ``xi_01`` error by a factor of ~30,
    because the product's variance is set by ``<phi_0^2><phi_1^2>``
    (O(1e-2)^2 at late times) while ``xi_01`` itself is only O(4e-4).

    ``sim_cache_a0.6.npz`` stores no sum of squares, so the variance is
    reconstructed from the field's own second moments.  For a zero-mean
    jointly Gaussian pair, Isserlis gives ``Var[XY] = <X^2><Y^2> +
    <XY>^2``, and ``<phi_a(0)^2> = xi_aa(r=0)``, which the cache does
    have.  The field is not exactly Gaussian, so this is an estimate.
    Measured against the 2M-realisation runs of
    ``paper_assets/demo2_kappa4/sim_dt_study.py``, which DO accumulate
    sums of squares: this estimator is 0.87-1.00 of the measured SEM
    over t in [0.6, 50] and r in {0, 0.5}, i.e. low by at most 13 %.
    The ``sqrt(xi^2 / n)`` it replaces is 0.029 of the measured SEM for
    ``xi_01`` (low by 34x) and 0.48-0.63 for ``xi_00`` (low by 1.6-2x).

    Args:
        series: the plotted ``xi_ab`` values.
        t_index: ``None`` when ``series`` runs over time at one
            separation; a time index when it runs over separation at
            one time.
    """
    a, b = pair
    xi = sim["xi"]
    v_a = xi[SIM_PAIR_IDX[(a, a)], :, 0]        # <phi_a(0)^2>(t)
    v_b = xi[SIM_PAIR_IDX[(b, b)], :, 0]        # <phi_b(0)^2>(t)
    if t_index is not None:
        v_a, v_b = v_a[t_index], v_b[t_index]
    var_prod = v_a * v_b + np.asarray(series) ** 2
    return np.sqrt(np.maximum(var_prod, 0.0) / sim["n_real"])


# =====================================================================
# Step 1: run the two L2 sweeps and unpack into pert[(a, b)] dicts.
# =====================================================================


def _check_blas_thread_caps() -> None:
    capped = any(
        os.environ.get(name)
        for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
    )
    if not capped:
        print(
            "[reproduce_figures] tip: configs set sweep.n_jobs=-1; "
            "set OPENBLAS_NUM_THREADS=1 / MKL_NUM_THREADS=1 / "
            "OMP_NUM_THREADS=1 to avoid BLAS oversubscription.",
            file=sys.stderr,
        )


def run_l2_sweep(config_path: Path, label: str):
    print(f"[L2/{label}] loading {config_path.name}")
    cfg = load_workflow_config(config_path)
    t0 = time.perf_counter()
    sweep, totals = run_workflow(cfg)
    print(f"[L2/{label}] done in {time.perf_counter() - t0:.1f}s "
          f"({len(totals)} totals rows)")
    return sweep, totals


def assemble_pert(totals_FF, totals_FK, sweep_FF, sweep_FK,
                  pert_t: np.ndarray, pert_r: np.ndarray) -> dict:
    """Slice the L2 sweep results into the notebook's
    ``pert[(a, b)] = {'0': (n_t, n_r), 'FF': ..., 'FK': ...}`` dict.
    """
    pert: dict = {pair: {} for pair in PAIRS}
    n_t, n_r = len(pert_t), len(pert_r)

    def _empty():
        return np.zeros((n_t, n_r))

    for a, b in PAIRS:
        pert[(a, b)] = {"0": _empty(), "FF": _empty(), "FK": _empty()}

    # FF totals carry both order 0 (free C) and order 2 (FF). Bin by order.
    # Cross-pair order-0 is identically zero (the package's
    # ``apply_diagonal`` retains the ``delta_{a, b}`` factor under
    # diag_C, so the C[a, b] propagator collapses to 0 when the
    # observable indices differ -- no hard-code needed).
    for _, row in totals_FF.iterrows():
        pair = (int(row["a"]), int(row["b"]))
        if pair not in pert:
            continue
        ti = int(np.argmin(np.abs(pert_t - row["t_final"])))
        ri = int(np.argmin(np.abs(pert_r - row["y"])))
        order = int(row["order"])
        value = float(row["value"])
        if order == 0:
            pert[pair]["0"][ti, ri] = value
        elif order == 2:
            pert[pair]["FF"][ti, ri] = value

    # FK totals are order 2 only (vertex_types: [FK]).
    for _, row in totals_FK.iterrows():
        pair = (int(row["a"]), int(row["b"]))
        if pair not in pert:
            continue
        ti = int(np.argmin(np.abs(pert_t - row["t_final"])))
        ri = int(np.argmin(np.abs(pert_r - row["y"])))
        pert[pair]["FK"][ti, ri] = row["value"]

    return pert


# =====================================================================
# Step 2: load the simulation cache (200K realisations, alpha=0.6).
# =====================================================================


def load_sim_cache():
    if not SIM_CACHE.exists():
        raise FileNotFoundError(
            f"missing {SIM_CACHE}. Run "
            f"``python ../run_simulation.py --alpha 0.6`` first."
        )
    raw = np.load(SIM_CACHE)
    return {
        "r": raw["r"],
        "t": raw["t"],
        "xi": raw["xi"],     # (3, n_t, n_r)
        "var0": raw["var0"], # (3, n_t)
        "mu2": raw["mu2"],   # (2, n_t)
        "mu3": raw["mu3"],   # (2, n_t)
        "n_real": int(raw["n_real"]),
        "alpha": float(raw["alpha"]),
    }


# =====================================================================
# Figure 1: kappa3_crosscheck.pdf -- pure simulation vs analytical
# =====================================================================


def figure_kappa3_crosscheck(sim: dict) -> None:
    print("[fig] kappa3_crosscheck.pdf")
    sim_alpha = sim["alpha"]
    sim_t = sim["t"]
    sim_mu2 = sim["mu2"]
    sim_mu3 = sim["mu3"]
    sim_n_real = sim["n_real"]

    # kappa3 = <eta_tilde^3> = 6 alpha lam^2 + 8 alpha^3 lam^3 (the alpha^3
    # term is the connected 3-point function of eta^2 - lam).
    k3_predict = 6.0 * sim_alpha * LAM ** 2 + 8.0 * sim_alpha ** 3 * LAM ** 3
    mu2_predict = LAM + 2.0 * sim_alpha ** 2 * LAM ** 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    for a in range(2):
        ax1.plot(sim_t, sim_mu2[a], "o-", label=fr"$\mu_2$ sim (a={a+1})")
    ax1.axhline(mu2_predict, color="k", ls="--",
                label=fr"$\lambda(1+2\alpha^2\lambda)$")
    ax1.set_xscale("log")
    ax1.set_xlabel("t")
    ax1.set_ylabel(r"$\mu_2$")
    ax1.legend()
    ax1.set_title(r"2nd moment of $\tilde\eta_a(0, t)$")

    for a in range(2):
        ax2.plot(sim_t, sim_mu3[a], "o-", label=fr"$\mu_3$ sim (a={a+1})")
    ax2.axhline(k3_predict, color="k", ls="--",
                label=fr"$6\alpha\lambda^2 + 8\alpha^3\lambda^3$")
    ax2.set_xscale("log")
    ax2.set_xlabel("t")
    ax2.set_ylabel(r"$\mu_3$")
    ax2.legend()
    ax2.set_title(r"3rd moment of $\tilde\eta_a(0, t)$")
    fig.suptitle(
        fr"$\kappa^{{(3)}}$ cross-check   "
        fr"($\alpha={sim_alpha}$, N={sim_n_real:,})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "kappa3_crosscheck.pdf", bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Figure 2: xi_vs_time.pdf -- 3 pairs at r=0, top + residual panels
# =====================================================================


def figure_xi_vs_time(pert: dict, sim: dict, pert_t: np.ndarray,
                      pert_r: np.ndarray) -> None:
    print("[fig] xi_vs_time.pdf")
    sim_t = sim["t"]
    sim_xi = sim["xi"]
    sim_n_real = sim["n_real"]
    sim_alpha = sim["alpha"]
    rr_target = 0.0
    ir = int(np.argmin(np.abs(pert_r - rr_target)))
    sim_ir = int(np.argmin(np.abs(sim["r"] - rr_target)))

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex="col")
    for col, (a, b) in enumerate(PAIRS):
        p = pert[(a, b)]
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        sim_series = sim_xi[SIM_PAIR_IDX[(a, b)], :, sim_ir]
        mc_err = mc_error(sim, (a, b), sim_series)

        curve0 = p["0"][:, ir]
        curve0FF = curve0 + p["FF"][:, ir]
        curve0FFFK = curve0FF + p["FK"][:, ir]

        ax_top.plot(pert_t, curve0, "--", color="grey", label="order 0")
        ax_top.plot(pert_t, curve0FF, "-.", color="tab:orange",
                    label="0 + FF")
        ax_top.plot(pert_t, curve0FFFK, "-", color="tab:red",
                    label="0 + FF + FK")
        ax_top.errorbar(sim_t, sim_series, yerr=mc_err, fmt="o", ms=4,
                        color="k", label=f"sim ({sim_n_real:,})",
                        zorder=5)
        ax_top.set_xscale("log")
        ax_top.set_title(fr"$\xi_{{{a}{b}}}(r={rr_target},t)$")
        if col == 0:
            ax_top.set_ylabel(r"$\xi$")
            ax_top.legend(fontsize=8)
        ax_top.grid(alpha=0.3)

        ax_bot.axhline(0, color="grey", lw=0.8)
        res_FF = sim_series - curve0FF
        res_FFFK = sim_series - curve0FFFK
        ax_bot.errorbar(sim_t, res_FF, yerr=mc_err, fmt="o-", ms=3,
                        color="tab:orange", label="sim $-$ (0+FF)")
        ax_bot.errorbar(sim_t, res_FFFK, yerr=mc_err, fmt="s-", ms=3,
                        color="tab:red", label="sim $-$ (0+FF+FK)")
        ax_bot.set_xlabel("t")
        if col == 0:
            ax_bot.set_ylabel("residual")
        ax_bot.set_xscale("log")
        if col == 1:
            ax_bot.legend(fontsize=8, loc="best")
        ax_bot.grid(alpha=0.3)

    fig.suptitle(
        fr"Demo 2   $\alpha={sim_alpha}$   ($\lambda=$ {LAM})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "xi_vs_time.pdf", bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Figure 3: xi_vs_r.pdf -- 3 pairs at 3 selected times
# =====================================================================


def figure_xi_vs_r(pert: dict, sim: dict, pert_t: np.ndarray,
                   pert_r: np.ndarray) -> None:
    print("[fig] xi_vs_r.pdf")
    sim_t = sim["t"]
    sim_r = sim["r"]
    sim_xi = sim["xi"]
    sim_n_real = sim["n_real"]
    sim_alpha = sim["alpha"]
    selected_t = [1.0, 5.0, 15.0]
    t_idx = [int(np.argmin(np.abs(sim_t - tt))) for tt in selected_t]
    pert_t_idx = [int(np.argmin(np.abs(pert_t - tt))) for tt in selected_t]

    fig, axes = plt.subplots(
        len(PAIRS), len(selected_t),
        figsize=(4 * len(selected_t), 3.5 * len(PAIRS)),
        sharex=True,
    )
    for row, (a, b) in enumerate(PAIRS):
        p = pert[(a, b)]
        for col, (it, pti) in enumerate(zip(t_idx, pert_t_idx)):
            ax = axes[row, col]
            sim_series = sim_xi[SIM_PAIR_IDX[(a, b)], it, :]
            mc_err = mc_error(sim, (a, b), sim_series, t_index=it)
            curve0 = p["0"][pti, :]
            curve0FF = curve0 + p["FF"][pti, :]
            curve0FFFK = curve0FF + p["FK"][pti, :]
            ax.plot(pert_r, curve0, "--", color="grey", label="0")
            ax.plot(pert_r, curve0FF, "-.", color="tab:orange",
                    label="0+FF")
            ax.plot(pert_r, curve0FFFK, "-", color="tab:red",
                    label="0+FF+FK")
            ax.errorbar(
                sim_r, sim_series, yerr=mc_err, fmt="o", ms=3,
                color="k",
                label=("sim" if (row == 0 and col == 0) else None),
            )
            ax.set_title(fr"$\xi_{{{a}{b}}}$, t={sim_t[it]:.2g}")
            if col == 0:
                ax.set_ylabel(r"$\xi$")
            if row == len(PAIRS) - 1:
                ax.set_xlabel("r")
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    fig.suptitle(
        fr"Demo 2  $\xi_{{ab}}(r)$ at selected t   "
        fr"($\alpha={sim_alpha}$)",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "xi_vs_r.pdf", bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Figure 4: xi_FK_only.pdf -- isolated FK channel
# =====================================================================


def figure_xi_FK_only(pert: dict, sim: dict, pert_t: np.ndarray,
                      pert_r: np.ndarray) -> None:
    print("[fig] xi_FK_only.pdf")
    sim_alpha = sim["alpha"]
    selected_t = [1.0, 5.0, 15.0]
    pert_t_idx = [int(np.argmin(np.abs(pert_t - tt))) for tt in selected_t]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    for col, (a, b) in enumerate(PAIRS):
        ax = axes[col]
        FK_mat = pert[(a, b)]["FK"]
        for it, tt in zip(pert_t_idx, selected_t):
            ax.plot(pert_r, FK_mat[it, :], label=fr"t={tt:.2g}")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(fr"FK contribution to $\xi_{{{a}{b}}}$")
        ax.set_xlabel("r")
        if col == 0:
            ax.set_ylabel(r"$\xi^{\text{FK}}$")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle(
        fr"Demo 2  FK only   ($\alpha={sim_alpha}$)  "
        r"—  non-zero only for the cross pair (0,1)",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "xi_FK_only.pdf", bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Driver
# =====================================================================


def main() -> None:
    _check_blas_thread_caps()

    sweep_FF, totals_FF = run_l2_sweep(CONFIG_FF, "FF")
    sweep_FK, totals_FK = run_l2_sweep(CONFIG_FK, "FK")

    # Pull the (r, t) grid from the FF totals; matches the YAMLs.
    pert_r = np.array(sorted(set(totals_FF["y"].tolist())))
    pert_t = np.array(sorted(set(totals_FF["t_final"].tolist())))
    print(f"[L2] grid: {len(pert_t)} t x {len(pert_r)} r")

    pert = assemble_pert(totals_FF, totals_FK, sweep_FF, sweep_FK,
                         pert_t, pert_r)

    sim = load_sim_cache()
    print(f"[sim] loaded {SIM_CACHE.name}: alpha={sim['alpha']}, "
          f"N={sim['n_real']:,}")

    figure_kappa3_crosscheck(sim)
    figure_xi_vs_time(pert, sim, pert_t, pert_r)
    figure_xi_vs_r(pert, sim, pert_t, pert_r)
    figure_xi_FK_only(pert, sim, pert_t, pert_r)

    print(f"\n[done] figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
