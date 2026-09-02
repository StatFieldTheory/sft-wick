r"""Paper figures for demo 3, from the caches written by ``level_a.py`` and
``level_b.py``.

Every comparison panel carries a residual sub-panel in units of the
Monte-Carlo error, because that is the only honest way to read agreement
when the reference itself is stochastic: a curve that "looks right" on a
log axis can be many sigma off, and one that looks off can be within one.

Run ``python level_a.py && python level_b.py && python make_figures.py``.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("SFT_WICK_QUIET_CACHE", "1")

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 10, "figure.titlesize": 13,
})

C_EXACT, C_PKG, C_SIM = "0.25", "tab:blue", "tab:red"


def _residual_axes(fig, gs_top, gs_bot, sharex=None):
    ax = fig.add_subplot(gs_top)
    axr = fig.add_subplot(gs_bot, sharex=ax)
    ax.tick_params(labelbottom=False)
    return ax, axr


def _pull_panel(axr, x, pull, xlabel):
    axr.axhline(0.0, color="0.5", lw=0.8)
    axr.axhspan(-1, 1, color="0.85", zorder=0)
    axr.axhspan(-2, 2, color="0.93", zorder=0)
    axr.plot(x, pull, "o-", color=C_SIM, ms=4)
    axr.set_ylabel("residual\n[MC $\\sigma$]")
    axr.set_xlabel(xlabel)
    axr.set_ylim(-4, 4)
    axr.grid(alpha=0.3)


def fig_level_a(d):
    """``⟨φ³⟩(t)``: closed form vs package vs event-exact simulation."""
    fig = plt.figure(figsize=(11, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.2], hspace=0.08, wspace=0.28)
    t, r = d["t_grid"], d["r_grid"]

    for col, (xg, exact, pkg, sim, err, xlabel, title) in enumerate([
        (t, d["exact3"], d["pkg3"], d["sim3"], d["err3"], "$t$",
         r"$\langle\phi(0,t)^3\rangle$  (all points coincident)"),
        (r, d["exactr"], d["pkgr"], d["simr"], d["errr"], "$r$",
         r"$\langle\phi(0,t)^2\phi(r,t)\rangle$,  $t=2$"),
    ]):
        ax, axr = _residual_axes(fig, gs[0, col], gs[1, col])
        ax.plot(xg, exact, "-", color=C_EXACT, lw=2, label="closed form")
        ax.plot(xg, pkg, "s", color=C_PKG, ms=7, mfc="none", mew=1.6,
                label="sft-wick")
        ax.errorbar(xg, sim, yerr=err, fmt="o", color=C_SIM, ms=4, capsize=2,
                    label="event-exact simulation")
        ax.set_ylabel(r"$\langle\phi^3\rangle$")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend()
        _pull_panel(axr, xg, (sim - exact) / err, xlabel)

    fig.suptitle(
        f"Level A ($F=0$): package vs closed form to "
        f"{max(np.abs(d['pkg3'] - d['exact3']).max() / np.abs(d['exact3']).max(), 1e-16):.0e} "
        f"relative; simulation has no $\\Delta t$ and no spatial discretisation error",
        y=1.02, fontsize=11)
    fig.savefig(FIG / "level_a_three_point.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_scaling(d):
    """The ``1/√n`` law: the non-Gaussian channel vanishes, ``κ₂`` does not."""
    n, exact, sim, err = d["n_sweep"], d["n_exact"], d["n_sim"], d["n_err"]
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(5.6, 5.0), height_ratios=[3, 1.2], sharex=True,
        gridspec_kw={"hspace": 0.08})
    nn = np.logspace(np.log10(n.min() / 1.6), np.log10(n.max() * 1.6), 100)
    ax.plot(nn, exact[np.argmin(np.abs(n - 1.0))] * np.sqrt(1.0 / nn), "-",
            color=C_EXACT, lw=2, label=r"$\propto 1/\sqrt{n}$")
    ax.plot(n, exact, "s", color=C_PKG, ms=8, mfc="none", mew=1.6,
            label="closed form")
    ax.errorbar(n, sim, yerr=err, fmt="o", color=C_SIM, ms=5, capsize=3,
                label="event-exact simulation")
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_ylabel(r"$\langle\phi^3\rangle$   at $t=1.5$")
    ax.set_title(r"Non-Gaussianity knob $n=\nu\sigma_t\sigma_x$"
                 "\n" r"($\kappa_2$ held fixed by compensating $h$)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    _pull_panel(axr, n, (sim - exact) / err, r"$n$")
    axr.set_xscale("log")
    fig.savefig(FIG / "level_a_scaling.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_level_b(d):
    """``ξ₀₁`` in time and separation, against ``Fκ³ + F³κ³ + F³κ⁵``."""
    fig = plt.figure(figsize=(11, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.2], hspace=0.08, wspace=0.28)
    t, r, s = d["t_grid"], d["r_grid"], float(d["s"])

    ax, axr = _residual_axes(fig, gs[0, 0], gs[1, 0])
    lead = s * d["th_fk3"]
    ax.plot(t, lead, "--", color="tab:orange", lw=1.6, label=r"$F\kappa^3$ only")
    ax.plot(t, d["total_t"], "-", color=C_EXACT, lw=2,
            label=r"$F\kappa^3+F^3\kappa^3+F^3\kappa^5$")
    ax.errorbar(t, d["sim_t"], yerr=d["sim_t_err"], fmt="o", color=C_SIM, ms=4,
                capsize=2, label="simulation (ETD, exact noise)")
    ax.set_ylabel(r"$\xi_{01}(t)$")
    ax.set_title(r"$\xi_{01}=\langle\phi_0\phi_1\rangle$ at $r=0$")
    ax.grid(alpha=0.3), ax.legend(loc="lower right")
    _pull_panel(axr, t, (d["sim_t"] - d["total_t"]) / d["sim_t_err"], "$t$")

    ax2, axr2 = _residual_axes(fig, gs[0, 1], gs[1, 1])
    ax2.plot(r, s * d["th_fk3_r"], "--", color="tab:orange", lw=1.6,
             label=r"$F\kappa^3$ only")
    ax2.plot(r, d["total_r"], "-", color=C_EXACT, lw=2, label="all computed channels")
    ax2.errorbar(r, d["sim_r"], yerr=d["sim_r_err"], fmt="o", color=C_SIM, ms=4,
                 capsize=2, label="simulation")
    ax2.set_ylabel(r"$\xi_{01}(r)$")
    ax2.set_title(fr"$\xi_{{01}}$ vs separation, $t={float(d['t_r']):.0f}$"
                  "\n(sites placed exactly at each $r$)")
    ax2.grid(alpha=0.3), ax2.legend()
    _pull_panel(axr2, r, (d["sim_r"] - d["total_r"]) / d["sim_r_err"], "$r$")

    fig.suptitle(
        rf"Level B, $F$ amplitude $s={s}$: $\xi_{{01}}$ is driven by the ODD "
        r"cumulants alone (all Gaussian channels cancel identically)",
        y=1.02, fontsize=11)
    fig.savefig(FIG / "level_b_xi01.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_channels(d):
    """Channel decomposition and the size of what is *not* computed."""
    t, s = d["t_grid"], float(d["s"])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    ax.plot(t, s * d["th_fk3"], "-o", ms=4, label=r"$F\kappa^3$ (order 2)")
    ax.plot(t, s ** 3 * d["th_f3k3"], "-s", ms=4, label=r"$F^3\kappa^3$ (order 4)")
    ax.plot(t, s ** 3 * d["th_f3k5"], "-^", ms=4,
            label=r"$F^3\kappa^5$ (order 4, ladder)")
    ax.set_yscale("log"), ax.set_xlabel("$t$"), ax.set_ylabel(r"$\xi_{01}$ channel")
    ax.set_title("Computed channels")
    ax.grid(alpha=0.3, which="both"), ax.legend()

    ratio = s ** 2 * d["th_f3k3"] / d["th_fk3"]
    ax2.plot(t, 100 * ratio, "-s", ms=4, color="tab:orange",
             label=r"$F^3\kappa^3/F\kappa^3$ (computed)")
    ax2.plot(t, 100 * s ** 2 * d["th_f3k5"] / d["th_fk3"], "-^", ms=4,
             color="tab:green", label=r"$F^3\kappa^5/F\kappa^3$ (computed)")
    ax2.plot(t, 100 * ratio ** 2, "--", color="0.4",
             label=r"$O(F^5)$ remainder (geometric estimate)")
    ax2.set_yscale("log"), ax2.set_xlabel("$t$")
    ax2.set_ylabel("per cent of the leading channel")
    ax2.set_title("Corrections, and what is left uncomputed")
    ax2.grid(alpha=0.3, which="both"), ax2.legend()
    fig.savefig(FIG / "level_b_channels.pdf", bbox_inches="tight")
    plt.close(fig)


def diagrams():
    """TikZ sources for the level-A and level-B diagrams."""
    from collections import OrderedDict
    from sft_wick.drawing_tikz import TikzRenderer
    import shot_noise as sn
    import system as dsys

    out = HERE / "diagrams"
    out.mkdir(exist_ok=True)
    tikz = TikzRenderer(standalone=False)
    written = []

    def dump(expansion, order, prefix, keep=None):
        """Render the unique diagrams of one order to TikZ.

        ``to_feynman_diagram`` raises a UID collision on the base commit
        (``ac7f201``) whenever a *non-local* vertex instance is allocated
        the same operator UIDs as the observable operators --- which is
        exactly what happens for level A's 3-point / order-1 / K3-only
        expansion (all 6 raw pairings fail).  The level-B F+K3 expansion
        is unaffected (42/42 render).  Numerics are untouched either way;
        only the *drawing* path needs the graph.  Skip and report rather
        than abort.
        """
        infos = expansion.raw_result.diagrams_by_order[order]
        groups, skipped = OrderedDict(), 0
        for info in infos:
            try:
                fd = info.to_feynman_diagram()
            except ValueError:
                skipped += 1
                continue
            groups.setdefault(fd.canonical_form(), []).append(fd)
        unique = [g[0] for g in groups.values()]
        k = 0
        for fd, dt in zip(unique, expansion.diagrams(order)):
            label = expansion._vertex_type_label(dt)
            if keep is not None and label != keep:
                continue
            k += 1
            (out / f"{prefix}_{k}.tex").write_text(tikz.to_string(fd))
            (out / f"{prefix}_{k}_standalone.tex").write_text(
                tikz.to_string(fd, standalone=True))
            written.append(f"{prefix}_{k}.tex  [{label}]  {dt.to_latex()}")
        if skipped:
            written.append(f"({skipped} pairing(s) under `{prefix}` could not be "
                           f"rendered: UID collision in to_feynman_diagram)")
        return k

    p = sn.PARAMS
    ex3 = dsys.make_system(p, cumulants=(3,)).expand(
        ("phi_a(x)", "phi_b(y)", "phi_c(z)"), orders=[1], progress=False)
    n_a = dump(ex3, 1, "level_a_kappa3")
    exb = dsys.make_system(p, f_amplitude=1.0, cumulants=(3,)).expand(
        ("phi_a(x)", "phi_b(y)"), orders=[2], progress=False)
    n_b = dump(exb, 2, "level_b_FK3", keep="FK3")
    (out / "README.md").write_text(
        "# Diagram sources\n\n"
        f"* `level_b_FK3_*.tex` -- the {n_b} order-2 F.kappa^(3) diagrams that\n"
        "  give the leading non-Gaussian signal in xi_01.\n"
        f"* `level_a_kappa3_*.tex` -- the level-A diagram ({n_a} rendered).\n"
        "  On the base commit ac7f201 the 3-point / order-1 / K3-only\n"
        "  expansion hits a UID collision inside `to_feynman_diagram`, so\n"
        "  the drawing is unavailable there. It affects rendering ONLY --\n"
        "  the level-A numbers agree with the closed form to 1e-16.\n\n"
        "Each has a `_standalone` twin that compiles on its own.\n\n"
        + "\n".join(f"* `{w}`" for w in written) + "\n")
    return n_a, n_b


def main():
    a = np.load(HERE / "level_a_results.npz")
    fig_level_a(a)
    fig_scaling(a)
    print("level A figures written")
    b_path = HERE / "level_b_results.npz"
    if b_path.exists():
        b = np.load(b_path)
        fig_level_b(b)
        fig_channels(b)
        print("level B figures written")
    else:
        print(f"skipping level B figures: {b_path.name} not found "
              f"(run `python level_b.py` first)")
    n_a, n_b = diagrams()
    print(f"diagrams: {n_a} level-A, {n_b} level-B FK3")
    for f in sorted(FIG.glob("*.pdf")):
        print(f"  {f.relative_to(HERE)}")


if __name__ == "__main__":
    main()
