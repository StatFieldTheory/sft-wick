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
t = B["t"]
r_th = B["r"]                 # theory grid (requested r + simulation grid sites)
r_sim = B["r_sim"]            # what the sim_* arrays are indexed by
r_sub = B["r_sub"]            # r values of the expensive sub-grid channels
x_grid = B["x_grid_sim"]      # the simulation's own spatial grid
t_check = B["t_check"]
PAIR_IDX = {"00": 0, "01": 1, "11": 2}
# Channels that make up the theory total, in table order.
CH_ORDER = ["o0_exact", "ff_exact", "fk", "ffk4", "fffk", "ffff"]
# Rows of the residual tables.  0.0 and 0.4 are simulation grid sites
# (dx = sigma_x / 5 = 0.2); 0.5 is off-grid and is the one the paper
# quotes, so it is kept -- with the theory interpolated exactly as the
# simulation interpolates (see ``interp_weights``).
R_ROWS = [0.0, 0.4, 0.5]


def interp_weights(r_val):
    """The two simulation grid sites ``np.interp`` blends at ``r_val``,
    and their weights.  Identity at a grid site.

    The simulation measures on ``x_grid`` and reports ``xi(r)`` as
    ``np.interp(r, x_grid, profile)``.  On a convex profile that is
    biased HIGH off-grid -- the observed +0.6-0.8 % excess in xi_00 at
    r = 0.25/0.5/0.75 versus +0.05-0.3 % at r = 0/0.4/1.0, which used to
    read as a "+3.7 sigma" physical residual.  Applying the SAME weights
    to the theory removes it exactly, because both sides are then the
    same linear functional of the same profile.
    """
    lo = int(np.clip(np.searchsorted(x_grid, r_val, side="right") - 1,
                     0, len(x_grid) - 2))
    hi = lo + 1
    w = (r_val - x_grid[lo]) / (x_grid[hi] - x_grid[lo])
    return (x_grid[lo], 1.0 - w), (x_grid[hi], w)


def theory(key, ch, r_val):
    """Theory channel ``ch`` for pair ``key`` at the separation the
    simulation reports as ``r_val``, interpolated the same way."""
    arr = B[f"{ch}_{key}"]
    grid = r_th if arr.shape[1] == len(r_th) else r_sub
    out = np.zeros(len(t))
    for site, w in interp_weights(r_val):
        if w == 0.0:
            continue
        i = int(np.argmin(np.abs(grid - site)))
        if abs(grid[i] - site) > 1e-9:
            raise KeyError(
                f"channel {ch} has no value at simulation grid site "
                f"{site} (its grid is {grid}); r = {r_val} cannot be "
                f"compared without biasing it by the interpolation."
            )
        out = out + w * arr[:, i]
    return out


def total(key, r_val, channels=CH_ORDER):
    return sum(theory(key, ch, r_val) for ch in channels)


def sim(key, r_val, which="sim_extrap"):
    ip = PAIR_IDX[key]
    ri = int(np.argmin(np.abs(r_sim - r_val)))
    assert abs(r_sim[ri] - r_val) < 1e-9, f"simulation has no r = {r_val}"
    return B[f"{which}_xi"][ip, :, ri], B[f"{which}_err"][ip, :, ri]


def fmt(x):
    return f"{x:.3e}"


def _fscale_section():
    """The D2 amplitude-scaling experiment, if it has been run."""
    path = HERE / "fscale_fit.json"
    if not path.exists():
        return ["## F-amplitude scaling (`fscale_fit.py`) -- NOT RUN", "",
                "Run `./run_fscale.sh` then `python fscale_fit.py`.", ""]
    f = json.loads(path.read_text())
    L = [f"## F-amplitude scaling: is the residual really order 4?", ""]
    L.append(
        "Scaling the quadratic drift by `s` scales each channel by a known "
        "power of `s` (FK ~ s, F³κ³ ~ s³, F⁵κ³ ~ s⁵), so "
        f"`residual(s) = xi_01^sim(s) - s·FK` should be `c3 s³ + c5 s⁵`.  "
        f"All amplitudes at dt = 0.02, at t = {f['t']:g}, r = {f['r']:g}: the "
        "step-size bias is common to all three and does not enter the "
        "s-dependence.  This is INDEPENDENT of the order-4 calculation -- "
        "it uses only the simulation and the validated order-2 channel.")
    L.append("")
    L.append("| s | runs | realisations | blow-ups /100k | xi_01 sim | s·FK | residual | ± MC | σ |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for q in f["rows"]:
        L.append(f"| {q['s']:.2f} | {q['n_files']} | {q['n_real']:,} | "
                 f"{q['blow_per_100k']:.1f} | {fmt(q['xi'])} | {fmt(q['fk'])} | "
                 f"{fmt(q['residual'])} | {q['err']:.1e} | {q['residual'] / q['err']:+.1f} |")
    L.append("")
    L.append("**s = 1.5 is outside the regime the expansion describes** and is "
             "not used for the fit.  Two independent signs of that, both in the "
             "table: the residual there EXCEEDS the leading term (1.04e-03 "
             "against s·FK = 5.16e-04), and 4.5 % of trajectories blow up in "
             "finite time against 6e-05 at s = 1 — a 780x jump.  The simulation "
             "then reports a mean conditioned on the survivors, i.e. with the "
             "largest excursions removed, which are precisely the realisations "
             "the higher-order terms describe.  It is kept in the table as a "
             "measured boundary of validity.")
    L.append("")
    L.append("| fit | amplitudes | model | chi² / dof | c3 | c5 |")
    L.append("|---|---|---|---|---|---|")
    L.append(f"| (a) | 0.5, 1.0, 1.5 | c3 s³ + c5 s⁵ | {f['chi2']:.2f} / {f['ndof']} | "
             f"{f['c3']:.3e} ± {f['c3_err']:.1e} | {f['c5']:.3e} ± {f['c5_err']:.1e} |")
    L.append(f"| **(b)** | **0.5, 1.0** | **c3 s³** | **{f['chi2_lowblowup']:.2f} / "
             f"{f['ndof_lowblowup']}** | **{f['c3_lowblowup']:.3e} ± "
             f"{f['c3_lowblowup_err']:.1e}** | — |")
    L.append("")
    L.append(f"Fit (b) is the one to read.  Its chi² of {f['chi2_lowblowup']:.2f} for "
             f"{f['ndof_lowblowup']} dof says the two clean amplitudes are consistent "
             f"with a **pure s³ law** — the residual is an order-4 effect.  That is a "
             f"deductive result: it uses only the simulation and the validated order-2 "
             f"channel, and assumes nothing about the order-4 calculation.  Fit (a) is "
             f"shown for completeness; its chi² of {f['chi2']:.2f} for {f['ndof']} dof "
             f"is the s = 1.5 point refusing to lie on any c3 s³ + c5 s⁵ curve through "
             f"the other two, which is the same statement as the paragraph above.")
    L.append("")
    L.append(f"Computed F³κ³ at s = 1: **{f['fffk_s1']:.3e}**, against fitted "
             f"c3 = {f['c3_lowblowup']:.3e} ± {f['c3_lowblowup_err']:.1e} — "
             f"**{f['pull_lowblowup']:+.1f}σ**.")
    L.append("")
    return L


def budget_table():
    L = []
    L.append("# Demo 2 error budget")
    L.append("")
    L.append(f"Parameters: alpha = {META['alpha']}, lambda = {META['lam']}, sigma_t = {META['sigma_t']}, "
             f"sigma_x = {META['sigma_x']}, gamma = {META['gamma']}; lam_eff = {META['lam_eff']:.4f}.")
    L.append("")
    L.append(f"**Simulation.** {META['n_real_sims']['0.02']:,} realisations at dt = 0.02 and "
             f"{META['n_real_sims']['0.01']:,} at dt = 0.01 ({META['n_files_sims']['0.02']} seeds each, "
             f"`sim_dt_study.py`), measured at exactly the theory times; "
             "'extrap' = Richardson (4 xi(0.01) - xi(0.02)) / 3 (Heun is O(dt^2)); the shipped cache "
             f"is {META['n_real_cache']:,} realisations at dt = {META['dt_cache']} on the nominal (unsnapped) times.")
    L.append("")
    L.append(f"**Separations.** The simulation measures on a grid of pitch "
             f"dx = sigma_x / 5 = {META['dx_sim']:g} and reports off-grid r by `np.interp`, "
             "which on a convex profile biases the value HIGH.  Every theory "
             "column below is interpolated with the SAME weights, so an "
             "off-grid row (r = 0.5) is compared like for like; r = 0.0 and "
             "r = 0.4 are grid sites and need no correction.  Before this fix "
             "the off-grid rows of xi_00 carried a spurious +0.6-0.8 % (+3.7 sigma) "
             "residual that was purely the interpolation.")
    L.append("")
    L.append("**Channels.** 0 = order 0 with the exact two-kernel C_eff; "
             "FF = order 2 F·F (exact C_eff); FK = order 2 F·κ³ (R-contracted, GL32); "
             "FFK4 = order 3 F·F·κ⁴ (R-contracted, GL12); "
             "**FFFK = order 4 F³·κ³ (R-contracted, GL10) — computed exactly for the "
             "first time in this revision; it was an equal-time estimate before, "
             "and was assumed to vanish for xi_00 / xi_11, which it does not**; "
             "FFFF = order 4 F⁴ (exact C_eff, Gauss-Legendre GL10 — it was 32768-sample "
             "Sobol QMC, which scattered 46 % across seeds).  "
             "FFFK and FFFF are computed on r_sub = " + str([float(x) for x in r_sub]) + " only.")
    L.append("")
    L.append("Theory wall-clock, "
             f"{'28' if META.get('n_jobs', -1) in (-1, 28) else META.get('n_jobs')} workers: "
             + ", ".join(f"{k} {v:.0f} s" for k, v in META["seconds"].items()) + ".")
    L.append("")
    for key in ("01", "00", "11"):
        for r_val in R_ROWS:
            sx, ex = sim(key, r_val)
            s2, e2 = sim(key, r_val, "sim_dt0.02")
            s1, e1 = sim(key, r_val, "sim_dt0.01")
            on_grid = min(abs(x_grid - r_val)) < 1e-9
            L.append(f"## xi_{key} at r = {r_val}"
                     + ("" if on_grid else "  (off-grid: theory interpolated to match)"))
            L.append("")
            L.append("| t | sim dt=.02 | sim dt=.01 | sim extrap ± err | 0 | 0: lam_eff − exact | FF | "
                     "FF: lam_eff − exact | FK | FFK4 | FFFK | FFFF | theory total | extrap − total | in σ |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            chans = {ch: theory(key, ch, r_val) for ch in CH_ORDER}
            d0 = theory(key, "o0_lameff", r_val) - chans["o0_exact"]
            dff = theory(key, "ff_lameff", r_val) - chans["ff_exact"]
            tot_all = sum(chans.values())
            for ti, tv in enumerate(t):
                res = sx[ti] - tot_all[ti]
                L.append(
                    f"| {tv:.3g} | {fmt(s2[ti])} ± {e2[ti]:.1e} | {fmt(s1[ti])} ± {e1[ti]:.1e} | "
                    f"{fmt(sx[ti])} ± {ex[ti]:.1e} | {fmt(chans['o0_exact'][ti])} | {d0[ti]:+.1e} | "
                    f"{fmt(chans['ff_exact'][ti])} | {dff[ti]:+.1e} | {fmt(chans['fk'][ti])} | "
                    f"{fmt(chans['ffk4'][ti])} | {fmt(chans['fffk'][ti])} | {fmt(chans['ffff'][ti])} | "
                    f"{fmt(tot_all[ti])} | {res:+.2e} | {res / ex[ti]:+.1f} |")
            chi2 = np.sum(((sx - tot_all) / ex) ** 2)
            L.append("")
            L.append(f"chi² of (extrap − total) over the {len(t)} times: {chi2:.1f}; "
                     f"mean pull {np.mean((sx - tot_all) / ex):+.2f}; "
                     f"largest |residual| {np.max(np.abs(sx - tot_all)):.2e}.")
            L.append("")

    L.append("## Truncation: F³κ³ (order 4), computed vs the estimate it replaces")
    L.append("")
    L.append("Until sft-wick 0.3.1 the package refused this channel "
             "(`NotImplementedError: Dynamic coupling with propagator-indexed "
             "contraction`), because a κ³ leg index survives onto a C propagator.  "
             "It was therefore ESTIMATED by collapsing κ³ to an equal-time constant "
             "24 α λ² σ_t² δ_abc and rescaling by (converged FK)/(collapsed FK) at "
             "the same t.  The estimate's calibration ratio is 0.42-0.64 for the "
             "FK-type partner-time configuration `(t', s, s)` but 1.08-1.50 for three "
             "distinct partner times, which is what the F³κ³ diagrams actually have — "
             "so it was a factor-of-2 quantity.  Both are now in the table:")
    L.append("")
    L.append("| t | FK converged | FK collapsed | ratio | FFFK collapsed | old ESTIMATE | **exact FFFK** | estimate/exact |")
    L.append("|---|---|---|---|---|---|---|---|")
    ri0 = int(np.argmin(np.abs(r_sub - 0.0)))
    for i, tv in enumerate(t_check):
        ti = int(np.argmin(np.abs(t - tv)))
        fk_exact = B["fk_01"][ti, int(np.argmin(np.abs(r_th - 0.0)))]
        fk_eq = B["fk_eq_01"][i, 0]
        ratio = fk_exact / fk_eq
        est = B["fffk_eq_01"][i, 0] * ratio
        exact = B["fffk_01"][ti, ri0]
        L.append(f"| {tv:.3g} | {fk_exact:.3e} | {fk_eq:.3e} | {ratio:.2f} | "
                 f"{B['fffk_eq_01'][i, 0]:.3e} | {est:.3e} | **{exact:.3e}** | {est / exact:.2f} |")
    L.append("")

    L.append("## Quadrature and Monte-Carlo error of each theory channel")
    L.append("")
    for ri, rv in enumerate(r_sub):
        rr = int(np.argmin(np.abs(r_th - rv)))
        L.append(f"- **FK_01, r = {rv}**: R-contracted GL32 vs GL64, max rel diff "
                 f"{np.max(np.abs(B['fk_01'][:, rr] - B['fk64_01'][:, ri]) / np.abs(B['fk64_01'][:, ri])):.1e}; "
                 f"raw-kernel GL8 (the pre-0.3.0 rule) / converged at t = 1, 3.48, 15, 50: "
                 + ", ".join(f"{B['fk_raw8_01'][int(np.argmin(np.abs(t - tv))), ri] / B['fk_01'][int(np.argmin(np.abs(t - tv))), rr]:.2f}"
                             for tv in (1.0, 3.48, 15.0, 50.0)))
    for i, tv in enumerate(t_check):
        ti = int(np.argmin(np.abs(t - tv)))
        a, b = B["ffk4_00"][ti, 0], B["ffk4_16_00"][i, 0]
        L.append(f"- **FFK4_00, r = 0, t = {tv:.3g}**: GL12 {a:.4e}, GL16 {b:.4e} "
                 f"(rel diff {abs(a - b) / max(abs(b), 1e-300):.1e}, abs {abs(a - b):.1e})")
    for key in ("00", "11", "01"):
        ref = B[f"ffff14_{key}"]
        if np.abs(ref).max() == 0.0:
            L.append(f"- **FFFF_{key}**: identically zero (F^4 is even in the "
                     f"noise; xi_01 is odd), by every rule tried.")
            continue
        d = np.abs(B[f"ffff_{key}"] - ref)
        rel = d / np.maximum(np.abs(ref), 1e-300)
        L.append(f"- **FFFF_{key}**: GL10 vs GL14 over the whole grid, "
                 f"max abs diff {d.max():.2e}, max rel {rel.max():.1e}.  That is "
                 f"the column's integration error; the residuals it is used to "
                 f"interpret are 3-6e-05, so it is "
                 f"{'well below them' if d.max() < 1e-5 else 'NOT below them'}.")
        dq = np.abs(B[f"ffff_qmc_{key}"] - ref)
        L.append(f"  - the superseded 32768-sample Sobol QMC differs from GL14 by "
                 f"up to {dq.max():.2e} absolute "
                 f"({(dq / np.maximum(np.abs(ref), 1e-300)).max():.0%} relative) — "
                 f"i.e. AS LARGE AS the residuals, which is why it had to go.")
    # FFFK: the 3-D rule loses the peak at large t_f exactly as the 4-D
    # FFFF rule does, so the node-count spread is quoted per time rather
    # than as one number over the grid.
    t_late = B["t_late"]
    a10, a8, a14 = B["fffk_01"], B["fffk8_01"], B["fffk14_01"]
    early = [i for i, tv in enumerate(t) if tv < t_late.min()]
    d_early = np.abs(a10[early] - a8[early]) / np.maximum(np.abs(a10[early]), 1e-300)
    L.append(f"- **FFFK_01**: GL8 vs GL10 agree to {d_early.max():.1%} for "
             f"t < {t_late.min():.3g} over all r.  At later times the 3-D rule "
             f"starts to lose the peak, so GL14 is the reference there "
             f"(r = 0, |GL14 - GL10| / GL14):")
    for j, tv in enumerate(t_late):
        i = int(np.argmin(np.abs(t - tv)))
        rel = abs(a14[j, 0] - a10[i, 0]) / abs(a14[j, 0])
        L.append(f"  - t = {tv:.4g}: GL8 {a8[i, 0]:.4e}, GL10 {a10[i, 0]:.4e}, "
                 f"GL14 {a14[j, 0]:.4e} — **{rel:.1%}**, "
                 f"{abs(a14[j, 0] - a10[i, 0]):.1e} absolute")
    L.append(f"  - the channel SATURATES: GL14 gives {a14[0, 0]:.3e} at "
             f"t = {t_late[0]:.3g} and {a14[-1, 0]:.3e} at t = {t_late[-1]:.3g}, "
             f"so the physical late-time value is ~{a14[:3, 0].mean():.2e} and the "
             f"slow rise across the last rows is quadrature, not physics.")
    i5 = int(np.argmin(np.abs(t - 5.44)))
    L.append(f"  - **xi_00 and xi_11 DO receive from this channel** — "
             f"{B['fffk_00'][i5, 0]:.3e} and {B['fffk_11'][i5, 0]:.3e} at "
             f"t = {t[i5]:.3g}, r = 0, converged to 0.1 % (GL8 vs GL10) — "
             f"and the 0.3.0 budget assumed they did not.  The assumption was "
             f"that phi_1 -> -phi_1 parity forbids odd cumulants there, but "
             f"that parity is BROKEN by the deformation itself: "
             f"eta~ = eta + alpha (eta^2 - lambda) is not odd in eta, which is "
             f"the whole reason xi_01 is non-zero.  The order-2 FK channel does "
             f"vanish for xi_00, but for the narrower reason that its two "
             f"diagrams' index structure does; that does not extend to order 4.  "
             f"This is worth {100 * B['fffk_00'][i5, 0] / 3.5e-5:.0f} % of the "
             f"xi_00 residual and was found only by computing all three pairs "
             f"instead of assuming.")
    L.append("")

    L.append("## Are the simulation error bars right?")
    L.append("")
    L.append("The quoted error is an inverse-variance combination of 20 "
             "independent seeds, each carrying its own per-realisation "
             "standard error.  Checked against the scatter of the seed means "
             "themselves (xi_01, r = 0): the ratio (seed-scatter error / "
             "quoted error) has median 1.06 at dt = 0.02 and 1.11 at "
             "dt = 0.01 over t >= 5, scattered on both sides of 1, and is "
             "0.63-0.74 at small t, i.e. if anything conservative there.  "
             "Per-t chi^2/dof across seeds is 0.34-1.62 throughout.")
    L.append("")
    L.append("One point looks alarming and is worth recording because the "
             "obvious statistic is the wrong one.  At dt = 0.02, t = 15 the "
             "plain seed scatter is **6.7x** the quoted error — while "
             "chi^2/dof at the same point is 0.83, which is contradictory "
             "unless one seed is a heavy-tailed outlier carrying a "
             "correspondingly large error.  It is: seed 105 gives "
             "1.7559e-03 +- 1.30e-03 against a median of 4.2397e-04 (54 MAD) "
             "— a near-blow-up trajectory.  The inverse-variance weighting "
             "downweights it automatically: dropping it moves the combined "
             "mean by **0.02 %** (4.16896e-04 -> 4.16816e-04) and brings the "
             "scatter ratio to 0.91.  So `std/sqrt(n)` over seeds is the "
             "wrong statistic on this data, not the errors; the `w = 1/e^2` "
             "combination is doing its job.")
    L.append("")
    L += _fscale_section()

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
    print("\n".join(L[:14]))


def fig_xi01_vs_time():
    key, r_val = "01", 0.0
    sx, ex = sim(key, r_val)
    fk = theory(key, "fk", r_val)
    fffk = theory(key, "fffk", r_val)
    raw8 = B["fk_raw8_01"][:, int(np.argmin(np.abs(r_sub - r_val)))]
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(6.4, 6.6), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1.4]})
    ax.plot(t, fk, "-", color="tab:red", label=r"FK: $F\times\kappa^{(3)}$ (R-contracted, converged)")
    ax.plot(t, fk + fffk, "-", color="tab:blue",
            label=r"FK + FFFK: $+\,F^3\times\kappa^{(3)}$ (order 4, exact)")
    ax.plot(t, raw8, ":", color="tab:gray", label=r"FK, raw kernel, 8-node tensor rule (paper v1)")
    ax.errorbar(t, sx, yerr=ex, fmt="o", ms=4, color="k",
                label=f"simulation, {META['n_real_sims']['0.01'] / 1e6:.0f}M realisations, $\\Delta t\\to0$", zorder=5)
    ax.set_xscale("log"); ax.set_ylabel(r"$\xi_{01}(r=0,t)$"); ax.grid(alpha=0.3); ax.legend(loc="lower right")
    ax.set_title(r"Demo 2: $\alpha=%g$, $\lambda=%g$, $\sigma_t=%g$ — the $\kappa^{(3)}$ channel" % (META["alpha"], META["lam"], META["sigma_t"]))
    ax.text(0.03, 0.93, "0, FF, FFFF, FFK4 vanish for $\\xi_{01}$\n($\\varphi_1\\to-\\varphi_1$ symmetry; only odd cumulants contribute)",
            transform=ax.transAxes, fontsize=8.5, va="top")
    axr.axhline(0, color="grey", lw=0.8)
    axr.errorbar(t, sx - fk, yerr=ex, fmt="s-", ms=3, color="tab:red", label="sim $-$ FK (converged)")
    axr.errorbar(t, sx - fk - fffk, yerr=ex, fmt="o-", ms=3, color="tab:blue",
                 label="sim $-$ (FK + FFFK)")
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
        # Theory on the simulation's OWN reported separations, with the
        # same np.interp weights, so the off-grid points are comparable.
        fk_r = np.array([theory(key, "fk", float(rv))[ti] for rv in r_sim])
        ax.plot(r_sim, fk_r, "-", color="tab:red", label="FK (R-contracted, converged)")
        ax.errorbar(r_sim, sx, yerr=ex, fmt="o", ms=4, color="k", label="simulation ($\\Delta t\\to0$)", zorder=5)
        ax.set_title(rf"$\xi_{{01}}(r,\,t={tv:.3g})$"); ax.set_xlabel("r"); ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$\xi_{01}$"); axes[0].legend()
    fig.tight_layout()
    fig.savefig(HERE / "xi01_vs_r.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_xi00_vs_time():
    key, r_val = "00", 0.0
    sx, ex = sim(key, r_val)
    o0, ff, fk, k4, fffk, f4 = (theory(key, ch, r_val) for ch in CH_ORDER)
    o0_l = theory(key, "o0_lameff", r_val); ff_l = theory(key, "ff_lameff", r_val)
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
