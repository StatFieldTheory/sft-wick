#!/usr/bin/env python
"""D2 -- the discriminating experiment: how does the residual scale with
the coupling amplitude?

The open claim in demo2 is that ``xi_01`` minus the converged order-2 FK
channel is the order-4 ``F^3 kappa^3`` truncation.  Order counting makes
that testable without any new theory: scaling the quadratic drift by
``s`` (``sim_dt_study.py --F_scale s``) scales every channel by a KNOWN
power of ``s``, because each channel contains a fixed number of F
vertices --

    order 0 : s^0      FK  : s^1      FF, FFK4 : s^2
    F^3.k^3 : s^3      FFFF: s^4      F^5.k^3  : s^5

so for ``xi_01`` (where order 0, FF, FFK4 and FFFF all vanish -- kappa^4
and the Gaussian channels are even under phi_1 -> -phi_1) the residual is

    residual(s) = xi_01^sim(s) - s * FK  =  c3 s^3 + c5 s^5 + O(s^7).

Fitting ``c3`` and comparing it with the F^3.kappa^3 that the package now
computes exactly is an INDEPENDENT check of the attribution: it uses only
the simulation and the (validated) order-2 channel, and it does not
assume the order-4 calculation is right.

All three amplitudes are compared at the SAME step size, dt = 0.02, so
the Heun O(dt^2) bias cancels out of the s-dependence rather than being
extrapolated away: the dt study measured xi_01(dt=0.02) - xi_01(dt=0.01)
= 7.5e-6 +- 4.9e-6 at t = 15, r = 0, well below the residual being fit.

Reads ``sims/`` (s = 1, the existing runs) and ``sims_fscale/`` (s = 0.5,
1.5).  Run ``./run_fscale.sh`` first.  Usage::

    python fscale_fit.py [--t 15.0] [--r 0.0]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
IDX = {(0, 0): 0, (0, 1): 1, (1, 1): 2}


def load_group(pattern):
    """Inverse-variance combine a set of runs; also total the blow-ups."""
    files = sorted(glob.glob(str(HERE / pattern)))
    if not files:
        raise FileNotFoundError(f"no runs matching {pattern}")
    xs, es, blown, attempted, n_real = [], [], 0, 0, 0
    t = r = None
    for f in files:
        d = np.load(f)
        if t is None:
            t, r = d["t"], d["r"]
        xs.append(d["xi"])
        es.append(d["xi_err"])
        n_real += int(d["n_real"])
        # Older runs (the shipped s = 1 set) predate the explicit
        # blow-up counters; n_real is n_good, and the request was a
        # round 100k, so the deficit is the blow-up count.
        if "n_attempted" in d:
            attempted += int(d["n_attempted"])
            blown += int(d["n_blown"])
        else:
            attempted += 100_000
            blown += 100_000 - int(d["n_real"])
    xs, es = np.array(xs), np.array(es)
    w = 1.0 / es ** 2
    return dict(t=t, r=r, xi=(w * xs).sum(0) / w.sum(0),
                err=1.0 / np.sqrt(w.sum(0)), n_real=n_real,
                n_files=len(files), n_blown=blown, n_attempted=attempted)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=float, default=15.0)
    ap.add_argument("--r", type=float, default=0.0)
    a = ap.parse_args()

    budget = np.load(HERE / "budget.npz")
    t_grid, r_grid = budget["t"], budget["r"]
    it = int(np.argmin(np.abs(t_grid - a.t)))
    ir = int(np.argmin(np.abs(r_grid - a.r)))
    ir_sub = int(np.argmin(np.abs(budget["r_sub"] - a.r)))
    fk1 = float(budget["fk_01"][it, ir])           # FK at s = 1
    fffk1 = float(budget["fffk_01"][it, ir_sub])   # F^3.kappa^3 at s = 1

    groups = {
        0.5: "sims_fscale/sim_F0.5_dt0.02_s*.npz",
        1.0: "sims/sim_dt0.02_s*.npz",
        1.5: "sims_fscale/sim_F1.5_dt0.02_s*.npz",
    }
    rows = []
    for s, pat in groups.items():
        g = load_group(pat)
        jt = int(np.argmin(np.abs(g["t"] - t_grid[it])))
        jr = int(np.argmin(np.abs(g["r"] - r_grid[ir])))
        xi = float(g["xi"][IDX[(0, 1)], jt, jr])
        err = float(g["err"][IDX[(0, 1)], jt, jr])
        rows.append(dict(s=s, xi=xi, err=err, fk=s * fk1,
                         residual=xi - s * fk1, n_real=g["n_real"],
                         n_files=g["n_files"], blow_per_100k=1e5 * g["n_blown"]
                         / max(g["n_attempted"], 1)))

    print(f"xi_01 at t = {t_grid[it]:g}, r = {r_grid[ir]:g}    "
          f"(FK(s=1) = {fk1:.4e}, exact F^3.k^3(s=1) = {fffk1:.4e})\n")
    print(f"{'s':>5} {'runs':>5} {'n_real':>11} {'blow/100k':>10} "
          f"{'xi_01 sim':>13} {'s*FK':>12} {'residual':>12} {'+- MC':>10} {'sigma':>7}")
    for q in rows:
        print(f"{q['s']:5.2f} {q['n_files']:5d} {q['n_real']:11,d} "
              f"{q['blow_per_100k']:10.1f} {q['xi']:13.5e} {q['fk']:12.5e} "
              f"{q['residual']:12.5e} {q['err']:10.2e} "
              f"{q['residual'] / q['err']:7.1f}")

    def wls(sel, powers):
        """Weighted least squares of ``residual`` on ``s**p`` for the
        selected amplitudes.  Returns (coeffs, sigmas, chi2, ndof)."""
        s_ = np.array([rows[i]["s"] for i in sel])
        y_ = np.array([rows[i]["residual"] for i in sel])
        e_ = np.array([rows[i]["err"] for i in sel])
        A = np.stack([s_ ** p for p in powers], axis=1) / e_[:, None]
        cov = np.linalg.inv(A.T @ A)
        c_ = cov @ (A.T @ (y_ / e_))
        model = sum(c_[k] * s_ ** p for k, p in enumerate(powers))
        return (c_, np.sqrt(np.diag(cov)), float(np.sum(((y_ - model) / e_) ** 2)),
                len(sel) - len(powers), cov)

    # (a) all three amplitudes, c3 and c5 free.
    c, sig, chi2, ndof, cov = wls([0, 1, 2], [3, 5])
    # (b) the two amplitudes whose blow-up fraction is negligible.  At
    #     s = 1.5 the simulation loses 4.5 % of its trajectories to
    #     finite-time blow-up and reports a mean conditioned on the
    #     survivors -- exactly the realisations the higher-order terms
    #     describe -- so that point is not a clean probe of the series.
    c2_, sig2_, chi2_2, ndof2, _ = wls([0, 1], [3])

    print(f"\n(a) all three amplitudes, residual(s) = c3 s^3 + c5 s^5   "
          f"(chi2 = {chi2:.2f}, ndof = {ndof})")
    print(f"      c3 = {c[0]:.4e} +- {sig[0]:.2e}")
    print(f"      c5 = {c[1]:.4e} +- {sig[1]:.2e}")
    print(f"      correlation(c3, c5) = {cov[0, 1] / (sig[0] * sig[1]):+.3f}")
    print(f"\n(b) s = 0.5 and 1.0 only (blow-up fractions 0 and 6e-5), "
          f"residual(s) = c3 s^3   (chi2 = {chi2_2:.2f}, ndof = {ndof2})")
    print(f"      c3 = {c2_[0]:.4e} +- {sig2_[0]:.2e}")
    print(f"\ncomparison with the exact order-4 calculation")
    print(f"  F^3.kappa^3 (computed)          = {fffk1:.4e}")
    pull = (c[0] - fffk1) / sig[0]
    pull2 = (c2_[0] - fffk1) / sig2_[0]
    print(f"  (a) c3 = {c[0]:.4e} +- {sig[0]:.2e}   ->  {pull:+.2f} sigma")
    print(f"  (b) c3 = {c2_[0]:.4e} +- {sig2_[0]:.2e}   ->  {pull2:+.2f} sigma")

    out = dict(t=float(t_grid[it]), r=float(r_grid[ir]), fk_s1=fk1,
               fffk_s1=fffk1, c3=float(c[0]), c3_err=float(sig[0]),
               c5=float(c[1]), c5_err=float(sig[1]), chi2=chi2, ndof=ndof,
               c3_lowblowup=float(c2_[0]), c3_lowblowup_err=float(sig2_[0]),
               chi2_lowblowup=chi2_2, ndof_lowblowup=ndof2,
               pull=float(pull), pull_lowblowup=float(pull2), rows=rows)
    (HERE / "fscale_fit.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {HERE / 'fscale_fit.json'}")


if __name__ == "__main__":
    main()
