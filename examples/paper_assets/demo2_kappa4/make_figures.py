#!/usr/bin/env python
"""Assemble the demo2 error budget from ``budget.npz``: the residual tables
(``budget.md``), the paper-ready figures and the FK diagrams in TikZ.

Figures (matplotlib rcParams as in ``examples/demo2/L2/reproduce_figures.py``):

* ``xi01_vs_time.pdf``: xi_01(r=0, t), simulation (2M realisations per
  step size, dt -> 0 extrapolated, Monte-Carlo errors) against FK, with
  a residual panel that also shows the paper's un-converged FK rule;
* ``xi01_vs_r.pdf``: xi_01(r) at two times;
* ``xi00_vs_time.pdf``: xi_00(r=0, t), simulation against
  0 + FF (+ FFK4 + FFFK + FFFF) with the exact C_eff, and the lam_eff
  approximation, with residual panel;
* ``fig_fk_diagrams.tex``: the two FK diagrams (TikZ), plus their
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
# quotes, so it is kept, with the theory interpolated exactly as the
# simulation interpolates (see ``interp_weights``).
R_ROWS = [0.0, 0.4, 0.5]


def interp_weights(r_val):
    """The two simulation grid sites ``np.interp`` blends at ``r_val``,
    and their weights.  Identity at a grid site.

    The simulation measures on ``x_grid`` and reports ``xi(r)`` as
    ``np.interp(r, x_grid, profile)``.  On a convex profile that is
    biased HIGH off-grid: the observed +0.6-0.8 % excess in xi_00 at
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
        return ["## F-amplitude scaling (`fscale_fit.py`): NOT RUN", "",
                "Run `./run_fscale.sh` then `python fscale_fit.py`.", ""]
    f = json.loads(path.read_text())
    L = [f"## F-amplitude scaling: is the residual really order 4?", ""]
    L.append(
        "Scaling the quadratic drift by `s` scales each channel by a known "
        "power of `s` (FK ~ s, F³κ³ ~ s³, F⁵κ³ ~ s⁵), so "
        f"`residual(s) = xi_01^sim(s) - s·FK` should be `c3 s³ + c5 s⁵`.  "
        f"All amplitudes at dt = 0.02, at t = {f['t']:g}, r = {f['r']:g}: the "
        "step size is common, but its bias can depend on the drift amplitude.   This is INDEPENDENT of the order-4 calculation: "
        "it uses only the simulation and the validated order-2 channel.")
    L.append("")
    L.append("| s | runs | realisations | blow-ups /100k | xi_01 sim | s·FK | residual | ± MC | σ |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for q in f["rows"]:
        L.append(f"| {q['s']:.2f} | {q['n_files']} | {q['n_real']:,} | "
                 f"{q['blow_per_100k']:.1f} | {fmt(q['xi'])} | {fmt(q['fk'])} | "
                 f"{fmt(q['residual'])} | {q['err']:.1e} | {q['residual'] / q['err']:+.1f} |")
    L.append("")
    L.append("**s = 1.5 is not a reliable point for testing this truncated series** "
             "and is excluded from fit (b).  Two warning signs are visible in the "
             "table: the residual there EXCEEDS the leading term (1.04e-03 "
             "against s·FK = 5.16e-04), and 4.5 % of trajectories blow up in "
             "finite time against 6e-05 at s = 1, a 780x jump.  The simulation "
             "then reports a mean conditioned on the survivors, with the "
             "largest excursions removed.  This changes the sampling population "
             "relative to the unconditioned perturbative moments.  It is kept "
             "in the table to document that limitation.")
    L.append("")
    L.append("| fit | amplitudes | model | chi² / dof | c3 | c5 |")
    L.append("|---|---|---|---|---|---|")
    L.append(f"| (a) | 0.5, 1.0, 1.5 | c3 s³ + c5 s⁵ | {f['chi2']:.2f} / {f['ndof']} | "
             f"{f['c3']:.3e} ± {f['c3_err']:.1e} | {f['c5']:.3e} ± {f['c5_err']:.1e} |")
    L.append(f"| **(b)** | **0.5, 1.0** | **c3 s³** | **{f['chi2_lowblowup']:.2f} / "
             f"{f['ndof_lowblowup']}** | **{f['c3_lowblowup']:.3e} ± "
             f"{f['c3_lowblowup_err']:.1e}** | n/a |")
    L.append("")
    L.append(f"Fit (b) is the one to read.  Its chi² of {f['chi2_lowblowup']:.2f} for "
             f"{f['ndof_lowblowup']} dof says the two low-blowup amplitudes are consistent "
             f"with a **pure s³ law**; two amplitudes do not uniquely establish a perturbative order.  The fit "
             f"uses only the simulation and the validated order-2 "
             f"channel, and assumes nothing about the order-4 calculation.  Fit (a) is "
             f"shown for completeness; its chi² of {f['chi2']:.2f} for {f['ndof']} dof "
             f"is the s = 1.5 point refusing to lie on any c3 s³ + c5 s⁵ curve through "
             f"the other two, which is the same statement as the paragraph above.")
    L.append("")
    L.append(f"Computed F³κ³ at s = 1: **{f['fffk_s1']:.3e}**, against fitted "
             f"c3 = {f['c3_lowblowup']:.3e} ± {f['c3_lowblowup_err']:.1e}, "
             f"**{f['pull_lowblowup']:+.1f}σ**.  The full cubic coefficient also "
             f"contains F³κ⁵, which is reported separately above.")
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
    L.append("**Channels.** 0 and FF use exact two-kernel C_eff (GL48); "
             "FK uses analytic R-contracted κ³ (GL64).  FFK4 = F²κ⁴, "
             "FFFK = F³κ³, and FFFF = F⁴ use refined Gauss-Legendre rules.  "
             "FFFK also requires exact C_eff.  Every refined L1 value is checked "
             "against independent Itô moment equations before it is saved.  "
             "The relative targets are 1e-4 for FFK4/FFFK and 1e-3 for FFFF; "
             "actual errors and node counts follow below.  These columns cover "
             "the listed channels, not the complete cumulant ladder.  "
             "All three component pairs receive FFFK contributions.  "
             "The higher-order channels use r_sub = " + str([float(x) for x in r_sub]) + ".")
    L.append("")
    L.append("Recorded stage wall-clock (legacy coarse diagnostics retain their original timings); new runs use "
             f"{META.get('n_jobs', 'unspecified')} workers: "
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
    L.append("Until sft-wick 0.4.0 the package refused this channel "
             "(`NotImplementedError: Dynamic coupling with propagator-indexed "
             "contraction`), because a κ³ leg index survives onto a C propagator.  "
             "It was therefore ESTIMATED by collapsing κ³ to an equal-time constant "
             "24 α λ² σ_t² δ_abc and rescaling by (converged FK)/(collapsed FK) at "
             "the same t.  The estimate's calibration ratio is 0.42-0.64 for the "
             "FK-type partner-time configuration `(t', s, s)` but 1.08-1.50 for three "
             "distinct partner times, which is what the F³κ³ diagrams actually have, "
             "so it was a factor-of-2 quantity.  Both are now in the table:")
    L.append("")
    L.append("| t | FK converged | FK collapsed | ratio | FFFK collapsed | old ESTIMATE | **checked FFFK** | estimate/checked |")
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

    L.append("## Numerical error against independent moment equations")
    L.append("")
    L.append("The reference uses a polynomial Markov generator for replicated "
             "OU noise.  Its F³h coefficient isolates κ³C and its F²h² "
             "coefficient isolates κ⁴, with h = M^(-1/2).  It imports no "
             "sft-wick code and uses no diagrams or propagator quadrature.  "
             "The values below remain actual package evaluations.  Differences "
             "between two coarse quadrature rules are diagnostics, not error bounds.")
    L.append("")
    L.append("| channel | pair | max absolute error | max relative error | GL nodes |")
    L.append("|---|---|---|---|---|")
    for channel in ("ffk4", "fffk", "ffff"):
        for key in ("00", "01", "11"):
            ref = B[f"ref_{channel}_{key}"]
            difference = np.abs(B[f"{channel}_{key}"] - ref)
            nonzero = ref != 0
            relative = np.max(difference[nonzero] / np.abs(ref[nonzero])) if nonzero.any() else 0
            nodes = B[f"{channel}_n_gauss"]
            L.append(f"| {channel.upper()} | {key} | {difference.max():.2e} | "
                     f"{relative:.2e} | {nodes.min()}–{nodes.max()} |")
    L.append("")
    L.append("| t, r=0 | FFFK GL8 | FFFK GL14 | checked L1 | moment reference | selected nodes |")
    L.append("|---|---|---|---|---|---|")
    for j, tv in enumerate(B["t_late"]):
        i = int(np.argmin(np.abs(t-tv)))
        L.append(f"| {tv:g} | {B['fffk8_01'][i,0]:.6e} | {B['fffk14_01'][j,0]:.6e} | "
                 f"{B['fffk_01'][i,0]:.6e} | {B['ref_fffk_01'][i,0]:.6e} | "
                 f"{B['fffk_n_gauss'][i]} |")
    L.append("")
    for key in ("00", "11"):
        ref = B[f"ref_ffff_{key}"]
        old = B[f"ffff14_{key}"]
        L.append(f"- Coarse FFFF_{key} GL14 differs from the independent reference by "
                 f"up to {np.max(np.abs(old-ref)/np.abs(ref)):.1%}.  "
                 f"It is not used as a converged reference for this budget.")
    ref5 = B["ref_fk5_01"][-1, 0]
    L.append(f"- The omitted F³κ⁵ channel is {ref5:.6e} at t={t[-1]:g}, r=0 "
             f"from the independent moment generator.  It is recorded as a "
             f"truncation check and is not included in the L1 theory total.")
    for ri, rv in enumerate(r_sub):
        rr = int(np.argmin(np.abs(r_th-rv)))
        difference = np.abs(B["fk_01"][:,rr]-B["fk96_01"][:,ri])
        L.append(f"- FK_01 at r={rv:g}: GL64/96 max relative difference "
                 f"{np.max(difference/np.abs(B['fk96_01'][:,ri])):.2e}.")
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
    L.append("One point illustrates the limits of this error-bar comparison.  "
             "At dt = 0.02, t = 15 the "
             "plain seed scatter is **6.7x** the quoted error, while "
             "chi^2/dof at the same point is 0.83.  That is contradictory "
             "unless one seed is a heavy-tailed outlier carrying a "
             "correspondingly large error, and it is: seed 105 gives "
             "1.7559e-03 +- 1.30e-03 against a median of 4.2397e-04 (54 MAD), "
             "a near-blow-up trajectory.  The inverse-variance weighting "
             "downweights it automatically: dropping it moves the combined "
             "mean by **0.02 %** (4.16896e-04 -> 4.16816e-04) and brings the "
             "scatter ratio to 0.91.  This explains why the weighted and "
             "unweighted summaries differ; it does not establish that the "
             "weighted uncertainty covers the unconditioned ensemble mean.  "
             "The estimated errors and means share trajectories, so "
             "inverse-variance weighting can suppress large positive "
             "excursions.  Heavy tails and discarded blow-ups remain "
             "limitations of this simulation comparison.")
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


def write_interpretation():
    """Keep numerical claims reproducible from the same saved budget."""
    total = sum(B[f"{channel}_01"][:, 0] for channel in CH_ORDER)
    simulation = B["sim_extrap_xi"][1, :, 0]
    uncertainty = B["sim_extrap_err"][1, :, 0]
    residual = simulation-total
    pull = residual/uncertainty
    leading_pull = (simulation-B["fk_01"][:, 0])/uncertainty
    i = int(np.argmin(np.abs(t-15)))
    rows = []
    for channel in ("ffk4", "fffk", "ffff"):
        for key in ("00", "01", "11"):
            ref = B[f"ref_{channel}_{key}"]
            difference = np.abs(B[f"{channel}_{key}"]-ref)
            nonzero = ref != 0
            relative = np.max(difference[nonzero]/np.abs(ref[nonzero])) if nonzero.any() else 0
            rows.append(f"| {channel.upper()} | {key} | {difference.max():.2e} | {relative:.2e} |")
    ledger = "\n".join(rows)
    text = f"""# Demo 2: verified channels and remaining limitations

Generated by `examples/paper_assets/demo2_kappa4/make_figures.py` from
the current `budget.npz`. The package values are independently checked
before the refined stages are saved.

## Physics and reference

The two fields obey `dphi0/dt = -phi0 + phi1² + eta_tilde0` and
`dphi1/dt = -phi1 + phi0*phi1 + eta_tilde1`, where
`eta_tilde = eta + alpha*(eta²-lambda)` and eta is stationary OU noise.
Parameters are alpha=0.6, lambda=0.05, gamma=1, sigma_t=0.3, sigma_x=1.
The fields start at zero. The observable is the fixed-time second moment,
not a moment of time-integrated fields.

The independent reference in `examples/reference/demo2_moments.py` uses
the polynomial Itô generator. For M independent noise replicas, normalized
by sqrt(M), covariance stays fixed while cumulant m scales as h^(m-2),
where h=M^(-1/2). The F³h coefficient isolates FFFK=F³κ³C;
F²h² isolates FFK4; F⁴h⁰ gives the Gaussian FFFF channel.
Two-site noise Gram variables carry the spatial correlations. This
reference imports no sft-wick code and uses no Wick diagrams or R/C quadrature.
It is checked through Gaussian, covariance, single-site and replica limits.

## What was wrong with the earlier budget

The September 12 budget's FFFK GL10/GL14 discrepancy reached 22.6% at t=50.
That was a difference between two unresolved rules, not an error bound.
At late times the integrand occupies a small part of the integration domain.
The budget also used bare-lambda C for FFFK, omitting the second term in
`C_eff = C[lambda*k] + C[2*alpha²*lambda²*k²]`. Its composite κ³ kernel
had additional inner quadrature error and did not declare partner-time
coincidences to the outer integrator. FF's QMC and FFFF's coarse GL rule
also lost accuracy at late times. These were pre-existing example-budget
issues; the T-001 C-table rewrite is bypassed by its closed-form C path.

The repaired κ³/κ⁴ callables integrate the exponential pieces analytically,
enforce leg causality, and declare their partner-time branches. FF now uses
GL48. FFK4, FFFK and FFFF refine actual package evaluations against the
independent reference. The relative acceptance targets are 1e-4, 1e-4 and
1e-3 respectively, on every saved nonzero cell. The reference does not
replace any L1 channel value.

## Validation ledger

Maximum errors over all saved times, spatial separations and the stated pair:

| channel | pair | max absolute error | max relative error |
|---|---|---|---|
{ledger}

At r=0,t={t[-1]:g}, the independent FFFK_01 reference is
**{B['ref_fffk_01'][-1,0]:.9e}**, and the saved package result is
**{B['fffk_01'][-1,0]:.9e}**. All three component pairs receive FFFK;
assuming its 00 and 11 entries vanish was incorrect.
Kernel tests additionally compare κ³ against raw adaptive triple integrals,
κ⁴ against randomized raw four-leg QMC, and check short-time, constant-kernel,
resonant-rate, permutation and cross-process serialization limits.

## Simulation residual and truncation

At r=0,t={t[i]:g}, the extrapolated simulation is
**{simulation[i]:.3e} ± {uncertainty[i]:.2e}**; the sum of the listed L1
channels is **{total[i]:.3e}**. The residual is
**{residual[i]:+.2e} ({pull[i]:+.1f}σ)**.
Adding FFFK changes χ² from {np.sum(leading_pull**2):.1f} to {np.sum(pull**2):.1f}
over the {len(t)} saved times at r=0; the mean pull is **{np.mean(pull):+.2f}**.
The χ² uses marginal errors; correlations between measurement times are
not accounted for, so it is a descriptive residual statistic.

The full deformed noise has cumulants beyond κ⁴. The independent generator
now computes the omitted F³κ⁵ contribution: **{B['ref_fk5_01'][-1,0]:.6e}**
at r=0,t={t[-1]:g}. It is saved as a truncation check, outside the listed
L1 total. Thus this total is not the complete order-four expansion of the
full cumulant ladder. Drift-amplitude fits at two low-blowup amplitudes
can support cubic scaling but cannot uniquely prove the source of a residual.

Numerical convergence of these channels does not establish convergence of
the perturbative series. The nonlinear simulation discards rare divergent
trajectories and therefore measures a conditioned mean. Finite sample size,
time-step extrapolation, higher perturbative orders and that conditioning
remain separate limitations. The current diagram/reference agreement
supports the tested scientific routes; it does not certify every possible
system or remove those modeling limitations.
"""
    (HERE.parents[1] / "demo2/INTERPRETATION.md").write_text(text)


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
            label=r"FK + FFFK: $+\,F^3\times\kappa^{(3)}$ (order 4, checked)")
    ax.plot(t, raw8, ":", color="tab:gray", label=r"FK, raw kernel, 8-node tensor rule (paper v1)")
    ax.errorbar(t, sx, yerr=ex, fmt="o", ms=4, color="k",
                label=f"simulation, {META['n_real_sims']['0.01'] / 1e6:.0f}M realisations, $\\Delta t\\to0$", zorder=5)
    ax.set_xscale("log"); ax.set_ylabel(r"$\xi_{01}(r=0,t)$"); ax.grid(alpha=0.3); ax.legend(loc="lower right")
    ax.set_title(r"Demo 2: $\alpha=%g$, $\lambda=%g$, $\sigma_t=%g$ the $\kappa^{(3)}$ channel" % (META["alpha"], META["lam"], META["sigma_t"]))
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
    listed_total = o0 + ff + fk + k4 + fffk + f4
    o0_l = theory(key, "o0_lameff", r_val); ff_l = theory(key, "ff_lameff", r_val)
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(6.4, 6.6), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1.4]})
    ax.plot(t, o0, "--", color="grey", label="order 0 (exact $C_{\\rm eff}$)")
    ax.plot(t, o0 + ff, "-.", color="tab:orange", label="0 + FF")
    ax.plot(t, listed_total, "-", color="tab:purple", label="0 + FF + FFK4 + FFFK + FFFF")
    ax.errorbar(t, sx, yerr=ex, fmt="o", ms=4, color="k",
                label=f"simulation, {META['n_real_sims']['0.01'] / 1e6:.0f}M realisations, $\\Delta t\\to0$", zorder=5)
    ax.set_xscale("log"); ax.set_ylabel(r"$\xi_{00}(r=0,t)$"); ax.grid(alpha=0.3); ax.legend(loc="lower right")
    ax.set_title(r"Demo 2: $\xi_{00}$, checked cumulant channels")
    axr.axhline(0, color="grey", lw=0.8)
    axr.errorbar(t, sx - (o0 + ff), yerr=ex, fmt="^-", ms=3, color="tab:orange", label="sim $-$ (0+FF), exact $C_{\\rm eff}$")
    axr.plot(t, sx - (o0_l + ff_l), "x:", color="tab:gray", label="sim $-$ (0+FF), $\\lambda_{\\rm eff}$ approximation (paper v1)")
    axr.errorbar(t, sx - listed_total, yerr=ex, fmt="o-", ms=3, color="tab:purple", label="sim $-$ (0+FF+FFK4+FFFK+FFFF)")
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
    write_interpretation()
    fig_xi01_vs_time()
    fig_xi01_vs_r()
    fig_xi00_vs_time()
    fk_diagrams()
