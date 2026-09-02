#!/usr/bin/env python
"""Assemble the demo2 error budget from ``budget.npz``: the residual tables
(``budget.md``), the paper-ready figures and the FK diagrams in TikZ.

Figures (matplotlib rcParams as in ``examples/demo2/L2/reproduce_figures.py``):

* ``xi01_vs_time.pdf`` -- xi_01(r=0, t): simulation (2M realisations per
  step size, dt -> 0 extrapolated, Monte-Carlo errors) against FK, with
  a residual panel that also shows the paper's un-converged FK rule;
* ``xi01_vs_r.pdf`` -- xi_01(r) at two times;
* ``xi00_vs_time.pdf`` -- xi_00(r=0, t): simulation against
  0 + FF (+ FFK4 + FFFF) with the exact C_eff, and the lam_eff
  approximation, with residual panel;
* ``fig_fk_diagrams.tex`` -- the two FK diagrams (TikZ), plus their
  DiagramTerm LaTeX in ``fk_diagrams.md``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "demo2"))

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 9, "figure.titlesize": 13,
})

B = dict(np.load(HERE / "budget.npz"))
META = json.loads((HERE / "budget_meta.json").read_text())
t, r, r_sub, t_check = B["t"], B["r"], B["r_sub"], B["t_check"]
PAIR_IDX = {"00": 0, "01": 1, "11": 2}
CH_ORDER = ["o0_exact", "ff_exact", "fk", "ffk4", "ffff"]


def col(key, ch, ri):
    arr = B[f"{ch}_{key}"]
    if arr.shape[1] == len(r):
        return arr[:, ri]
    return arr[:, list(r_sub).index(float(r[ri]))]


def total(key, ri, channels=CH_ORDER):
    return sum(col(key, ch, ri) for ch in channels)


def sim(key, ri, which="sim_extrap"):
    ip = PAIR_IDX[key]
    return B[f"{which}_xi"][ip, :, ri], B[f"{which}_err"][ip, :, ri]


def fmt(x):
    return f"{x:.3e}"


def budget_table():
    L = []
    L.append("# Demo 2 error budget")
    L.append("")
    L.append(f"Parameters: alpha = {META['alpha']}, lambda = {META['lam']}, sigma_t = {META['sigma_t']}, "
             f"sigma_x = {META['sigma_x']}, gamma = {META['gamma']}; lam_eff = {META['lam_eff']:.4f}.")
    L.append(f"Simulation: {META['n_real_sims']['0.02']:,} realisations at dt = 0.02 and "
             f"{META['n_real_sims']['0.01']:,} at dt = 0.01 ({META['n_files_sims']['0.02']} seeds each, "
             f"`sim_dt_study.py`), measured at exactly the theory times; "
             "'extrap' = Richardson (4 xi(0.01) - xi(0.02)) / 3 (Heun is O(dt^2)); the shipped cache "
             f"is {META['n_real_cache']:,} realisations at dt = {META['dt_cache']} on the nominal (unsnapped) times.")
    L.append("Theory wall-clock, 28 workers: " + ", ".join(f"{k} {v:.0f} s" for k, v in META["seconds"].items()) + ".")
    L.append("")
    L.append("Channels: 0 = order 0 with the exact two-kernel C_eff; FF = order 2 F·F (exact C_eff); "
             "FK = order 2 F·κ³ (R-contracted, converged); FFK4 = order 3 F·F·κ⁴ (R-contracted); "
             "FFFF = order 4 (exact C_eff, r = 0 and 0.5 only).  'lam_eff − exact' columns give what the "
             "single-kernel approximation used so far adds.")
    L.append("")
    for key in ("01", "00", "11"):
        for r_val in (0.0, 0.5):
            ri = int(np.argmin(np.abs(r - r_val)))
            sx, ex = sim(key, ri)
            s2, e2 = sim(key, ri, "sim_dt0.02")
            s1, e1 = sim(key, ri, "sim_dt0.01")
            L.append(f"## xi_{key} at r = {r_val}")
            L.append("")
            L.append("| t | sim dt=.02 | sim dt=.01 | sim extrap ± err | 0 | 0: lam_eff − exact | FF | "
                     "FF: lam_eff − exact | FK | FFK4 | FFFF | theory total | extrap − total | in σ |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for ti, tv in enumerate(t):
                o0, ff, fk, k4, f4 = (col(key, ch, ri)[ti] for ch in CH_ORDER)
                d0 = col(key, "o0_lameff", ri)[ti] - o0
                dff = col(key, "ff_lameff", ri)[ti] - ff
                tot = o0 + ff + fk + k4 + f4
                res = sx[ti] - tot
                L.append(f"| {tv:.3g} | {fmt(s2[ti])} ± {e2[ti]:.1e} | {fmt(s1[ti])} ± {e1[ti]:.1e} | "
                         f"{fmt(sx[ti])} ± {ex[ti]:.1e} | {fmt(o0)} | {d0:+.1e} | {fmt(ff)} | {dff:+.1e} | "
                         f"{fmt(fk)} | {fmt(k4)} | {fmt(f4)} | {fmt(tot)} | {res:+.2e} | {res / ex[ti]:+.1f} |")
            # chi^2 summary
            tot_all = total(key, ri)
            chi2 = np.sum(((sx - tot_all) / ex) ** 2)
            L.append("")
            L.append(f"chi² of (extrap − total) over the {len(t)} times: {chi2:.1f}; "
                     f"mean pull {np.mean((sx - tot_all) / ex):+.2f}; "
                     f"largest |residual| {np.max(np.abs(sx - tot_all)):.2e}.")
            L.append("")
    L.append("## Quadrature checks")
    L.append("")
    for ri, rv in enumerate(r_sub):
        rr = int(np.argmin(np.abs(r - rv)))
        L.append(f"- FK_01 at r = {rv}: R-contracted GL32 vs GL64, max rel diff "
                 f"{np.max(np.abs(B['fk_01'][:, rr] - B['fk64_01'][:, ri]) / np.abs(B['fk64_01'][:, ri])):.1e}; "
                 f"raw-kernel GL8 (the paper's rule) / converged at t = 1, 3.48, 15, 50: "
                 + ", ".join(f"{B['fk_raw8_01'][int(np.argmin(np.abs(t - tv))), ri] / B['fk_01'][int(np.argmin(np.abs(t - tv))), rr]:.2f}"
                             for tv in (1.0, 3.48, 15.0, 50.0)))
    for i, tv in enumerate(t_check):
        ti = int(np.argmin(np.abs(t - tv)))
        a, b = B["ffk4_00"][ti, 0], B["ffk4_16_00"][i, 0]
        L.append(f"- FFK4_00 at r = 0, t = {tv:.3g}: GL12 {a:.4e}, GL16 {b:.4e} (rel diff {abs(a - b) / max(abs(b), 1e-300):.1e})")
    L.append("")
    L.append("## Truncation: F³κ³ (order 4) estimate for xi_01 at r = 0")
    L.append("")
    L.append("Cannot go through the dynamic-coupling path (a κ³ index sits on a C propagator); "
             "estimated with κ³ collapsed to an equal-time constant, 24 α λ² σ_t² δ_abc, rescaled by the "
             "ratio (converged FK) / (collapsed FK) at the same t:")
    L.append("")
    for i, tv in enumerate(t_check):
        ti = int(np.argmin(np.abs(t - tv)))
        fk_exact = B["fk_01"][ti, 0]
        fk_eq = B["fk_eq_01"][i, 0]
        ratio = fk_exact / fk_eq
        est = B["fffk_eq_01"][i, 0] * ratio
        L.append(f"- t = {tv:.3g}: FK converged {fk_exact:.3e}, collapsed {fk_eq:.3e} (ratio {ratio:.2f}); "
                 f"FFFK collapsed {B['fffk_eq_01'][i, 0]:.3e} → estimate {est:.3e} "
                 f"({100 * est / fk_exact:.0f} % of FK)")
    L.append("")
    L.append("## Noise cumulants at x = 0 (simulation dt = 0.01, all times and seeds, vs analytic)")
    L.append("")
    from k4_coupling import single_site_cumulants
    k2, k3, k4 = single_site_cumulants()
    mu = B["sim_dt0.01_mu"]          # (3, 2, n_t)
    mu2, mu3, mu4 = (mu[i].mean() for i in range(3))
    L.append(f"- kappa2: sim {mu2:.5e}, analytic λ + 2α²λ² = {k2:.5e}")
    L.append(f"- kappa3: sim {mu3:.5e}, analytic 6αλ² + 8α³λ³ = {k3:.5e} (6αλ² alone = {6 * META['alpha'] * META['lam'] ** 2:.5e})")
    L.append(f"- kappa4: sim {mu4 - 3 * mu2 ** 2:.5e}, analytic 48α²λ³ + 48α⁴λ⁴ = {k4:.5e}")
    (HERE / "budget.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:12]))


def fig_xi01_vs_time():
    key, ri = "01", 0
    sx, ex = sim(key, ri)
    fk = col(key, "fk", ri)
    raw8 = B["fk_raw8_01"][:, 0]
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(6.4, 6.6), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1.4]})
    ax.plot(t, fk, "-", color="tab:red", label=r"FK: $F\times\kappa^{(3)}$ (R-contracted, converged)")
    ax.plot(t, raw8, ":", color="tab:gray", label=r"FK, raw kernel, 8-node tensor rule (paper v1)")
    ax.errorbar(t, sx, yerr=ex, fmt="o", ms=4, color="k",
                label=f"simulation, {META['n_real_sims']['0.01'] / 1e6:.0f}M realisations, $\\Delta t\\to0$", zorder=5)
    ax.set_xscale("log"); ax.set_ylabel(r"$\xi_{01}(r=0,t)$"); ax.grid(alpha=0.3); ax.legend(loc="lower right")
    ax.set_title(r"Demo 2: $\alpha=%g$, $\lambda=%g$, $\sigma_t=%g$ — the $\kappa^{(3)}$ channel" % (META["alpha"], META["lam"], META["sigma_t"]))
    ax.text(0.03, 0.93, "0, FF, FFFF, FFK4 vanish for $\\xi_{01}$\n($\\varphi_1\\to-\\varphi_1$ symmetry; only odd cumulants contribute)",
            transform=ax.transAxes, fontsize=8.5, va="top")
    axr.axhline(0, color="grey", lw=0.8)
    axr.errorbar(t, sx - fk, yerr=ex, fmt="s-", ms=3, color="tab:red", label="sim $-$ FK (converged)")
    axr.plot(t, sx - raw8, "x:", color="tab:gray", label="sim $-$ FK (8-node rule)")
    axr.set_xscale("log"); axr.set_xlabel("t"); axr.set_ylabel("residual"); axr.grid(alpha=0.3)
    axr.legend(loc="best")
    fig.tight_layout()
    fig.savefig(HERE / "xi01_vs_time.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_xi01_vs_r():
    key = "01"
    times = [3.48, 15.0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, tv in zip(axes, times):
        ti = int(np.argmin(np.abs(t - tv)))
        ip = PAIR_IDX[key]
        sx = B["sim_extrap_xi"][ip, ti, :]; ex = B["sim_extrap_err"][ip, ti, :]
        ax.plot(r, B["fk_01"][ti, :], "-", color="tab:red", label="FK (R-contracted, converged)")
        ax.errorbar(r, sx, yerr=ex, fmt="o", ms=4, color="k", label="simulation ($\\Delta t\\to0$)", zorder=5)
        ax.set_title(rf"$\xi_{{01}}(r,\,t={tv:.3g})$"); ax.set_xlabel("r"); ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$\xi_{01}$"); axes[0].legend()
    fig.tight_layout()
    fig.savefig(HERE / "xi01_vs_r.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_xi00_vs_time():
    key, ri = "00", 0
    sx, ex = sim(key, ri)
    o0, ff, fk, k4, f4 = (col(key, ch, ri) for ch in CH_ORDER)
    o0_l = col(key, "o0_lameff", ri); ff_l = col(key, "ff_lameff", ri)
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(6.4, 6.6), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1.4]})
    ax.plot(t, o0, "--", color="grey", label="order 0 (exact $C_{\\rm eff}$)")
    ax.plot(t, o0 + ff, "-.", color="tab:orange", label="0 + FF")
    ax.plot(t, o0 + ff + k4 + f4, "-", color="tab:purple", label="0 + FF + FFK4 ($\\kappa^{(4)}$) + FFFF")
    ax.errorbar(t, sx, yerr=ex, fmt="o", ms=4, color="k",
                label=f"simulation, {META['n_real_sims']['0.01'] / 1e6:.0f}M realisations, $\\Delta t\\to0$", zorder=5)
    ax.set_xscale("log"); ax.set_ylabel(r"$\xi_{00}(r=0,t)$"); ax.grid(alpha=0.3); ax.legend(loc="lower right")
    ax.set_title(r"Demo 2: $\xi_{00}$ — even cumulants")
    axr.axhline(0, color="grey", lw=0.8)
    axr.errorbar(t, sx - (o0 + ff), yerr=ex, fmt="^-", ms=3, color="tab:orange", label="sim $-$ (0+FF), exact $C_{\\rm eff}$")
    axr.plot(t, sx - (o0_l + ff_l), "x:", color="tab:gray", label="sim $-$ (0+FF), $\\lambda_{\\rm eff}$ approximation (paper v1)")
    axr.errorbar(t, sx - (o0 + ff + k4 + f4), yerr=ex, fmt="o-", ms=3, color="tab:purple", label="sim $-$ (0+FF+FFK4+FFFF)")
    axr.set_xscale("log"); axr.set_xlabel("t"); axr.set_ylabel("residual"); axr.grid(alpha=0.3)
    axr.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "xi00_vs_time.pdf", bbox_inches="tight")
    plt.close(fig)


def fk_diagrams():
    from collections import OrderedDict
    import sft_wick as sw
    from sft_wick.drawing_tikz import TikzRenderer
    from k3_R_coupling import coupling_fn_vectorized as k3_fn
    F = np.zeros((2, 2, 2)); F[0, 1, 1] = 1.0; F[1, 0, 1] = F[1, 1, 0] = 0.5
    system = sw.System(
        field=sw.FieldSpec("phi", 2), linear=sw.DiagonalA(gamma=[1.0, 1.0]),
        vertices=[sw.LocalVertex("F", F)],
        nonlocal_vertices=[sw.NonLocalVertex("K", 3, coupling=k3_fn, coupling_vectorized=True)],
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(0.05, 0.3), spatial=sw.ExponentialSpatial(1.0))),
    )
    exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2], progress=False)
    dts = exp.diagrams(2)
    infos = exp.raw_result.diagrams_by_order[2]
    groups = OrderedDict()
    for info in infos:
        fd = info.to_feynman_diagram()
        groups.setdefault(fd.canonical_form(), []).append(fd)
    unique = [g[0] for g in groups.values()]
    assert len(unique) == len(dts), (len(unique), len(dts))
    tikz = TikzRenderer(standalone=False)
    parts, md = [], ["# The FK diagrams (order 2, F x kappa^(3))", ""]
    k = 0
    for fd, dt in zip(unique, dts):
        if exp._vertex_type_label(dt) != "FK":
            continue
        k += 1
        src = tikz.to_string(fd)
        (HERE / f"fk_diagram_{k}.tex").write_text(src)
        (HERE / f"fk_diagram_{k}_standalone.tex").write_text(tikz.to_string(fd, standalone=True))
        parts.append(f"% FK diagram {k}\n" + src)
        md.append(f"{k}. propagators `{' '.join(p.to_latex() for p in dt.propagators)}`  ")
        md.append(f"   $$ {dt.to_latex()} $$")
        md.append("")
    (HERE / "fig_fk_diagrams.tex").write_text("\n".join(parts))
    (HERE / "fk_diagrams.md").write_text("\n".join(md) + "\n")
    print(f"{k} FK diagrams written")


if __name__ == "__main__":
    budget_table()
    fig_xi01_vs_time()
    fig_xi01_vs_r()
    fig_xi00_vs_time()
    fk_diagrams()
