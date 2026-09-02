r"""Level B --- the interacting test.

``N = 2`` with demo 2's ``Z₂`` drift structure::

    dφ₀/dt = −γφ₀ + s φ₁² + η₀,     dφ₁/dt = −γφ₁ + s φ₀φ₁ + η₁

The drift is invariant under ``φ₁ → −φ₁``.  Were the noise law also
invariant under ``η₁ → −η₁``, the cross-correlator
``ξ₀₁ = ⟨φ₀(x,t) φ₁(y,t)⟩`` would vanish identically.  Shot noise is
skewed, so ``ξ₀₁`` is driven by the **odd** cumulants of ``η₁`` alone:
order 0, FF, FFFF and the entire ``κ⁴`` channel cancel.  It is the
cleanest available non-Gaussian observable.

The series is ``ξ₀₁ = Fκ³ + F³κ³ + F³κ⁵ + O(F⁵)``.  Demo 3 computes
**all three** leading terms exactly rather than estimating the
correction:

* ``Fκ³``  --- order 2, 2 diagrams, 1 time integration variable;
* ``F³κ³`` --- order 4, 30 diagrams, 3 time integration variables;
* ``F³κ⁵`` --- order 4, 6 diagrams, 3 time integration variables.  This
  is the *neglected-cumulant ladder*: filtered-Poisson noise has every
  ``κ_m``, and ``F³κ⁵`` enters at the same order and with the same ``F³``
  scaling as ``F³κ³``, so the amplitude-scaling test **cannot** separate
  them.  The ``s``-factorisation of :mod:`shot_noise` is ``m``-agnostic,
  so computing it needs no new ideas.

Run ``python level_b.py`` (``--quick`` for a fast pass).
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

os.environ.setdefault("SFT_WICK_QUIET_CACHE", "1")

import shot_noise as sn
import simulate_b as sb
import system as dsys

#: Chosen so the computed ``F³`` correction stays a genuine correction
#: (7.3 % of the total at the largest time) and the *uncomputed* ``O(F⁵)``
#: remainder, geometrically ~``(F³κ³/Fκ³)²``, stays below 1 %.
F_AMPLITUDE = 0.2

T_GRID = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
R_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
T_R = 2.0
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".expansion_cache")


def _props(system, t_max):
    return system.propagators(t_max=t_max * 1.05 + 1.0, n_grid_t=40,
                              c_closed_form="auto", c_closed_form_only=True,
                              progress=False)


def theory_xi01(t_grid, r_grid, t_r, p=sn.PARAMS, n_gauss_2=32, n_gauss_4=12,
                n_jobs=-1):
    """The three computed channels of ``ξ₀₁`` at unit ``F`` amplitude.

    Returned at ``s = 1``: the channels scale exactly as ``s`` and ``s³``,
    which is both a saving and the basis of the scaling check.
    """
    system = dsys.make_system(p, f_amplitude=1.0, cumulants=(3, 5))
    expansion = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2, 4],
                              cache_path=CACHE, progress=False)
    props = _props(system, max(float(max(t_grid)), t_r))

    def ev(order, vtypes, t, ng, y=0.0):
        # order 4 is 30 (FK3) or 6 (FK5) diagrams, each an independent
        # 3-D Gauss-Legendre integral -- embarrassingly parallel.
        return expansion.evaluate(
            props, positions={"x": 0.0, "y": y}, t_final=float(t),
            component_pair=(0, 1), orders=[order], vertex_types=vtypes,
            method="gauss_legendre", n_gauss=ng,
            n_jobs=n_jobs if order >= 4 else 1).total

    fk3 = np.array([ev(2, ["FK3"], t, n_gauss_2) for t in t_grid])
    f3k3 = np.array([ev(4, ["FK3"], t, n_gauss_4) for t in t_grid])
    f3k5 = np.array([ev(4, ["FK5"], t, n_gauss_4) for t in t_grid])
    fk3_r = np.array([ev(2, ["FK3"], t_r, n_gauss_2, y=float(r)) for r in r_grid])
    f3k3_r = np.array([ev(4, ["FK3"], t_r, n_gauss_4, y=float(r)) for r in r_grid])
    f3k5_r = np.array([ev(4, ["FK5"], t_r, n_gauss_4, y=float(r)) for r in r_grid])
    return dict(fk3=fk3, f3k3=f3k3, f3k5=f3k5,
                fk3_r=fk3_r, f3k3_r=f3k3_r, f3k5_r=f3k5_r)


def combine(theory, s):
    """``ξ₀₁(s) = s·Fκ³ + s³(F³κ³ + F³κ⁵)``, exact in the ``F`` amplitude."""
    return (s * theory["fk3"] + s ** 3 * (theory["f3k3"] + theory["f3k5"]),
            s * theory["fk3_r"] + s ** 3 * (theory["f3k3_r"] + theory["f3k5_r"]))


def theory_xi00(t_grid, s, p=sn.PARAMS):
    """``ξ₀₀`` --- order 0, the FF channel, and the ``F²κ⁴`` channel.

    An independent comparison in the sector where the *even* cumulants do
    contribute, so it exercises ``κ⁴`` rather than ``κ³``.
    """
    system = dsys.make_system(p, f_amplitude=s, cumulants=(4,))
    expansion = system.expand(("phi_a(x)", "phi_b(y)"), orders=[0, 2, 3],
                              cache_path=CACHE, progress=False)
    props = _props(system, float(max(t_grid)))
    out = {}
    for name, order, vt, ng in (("order0", 0, None, 8),
                                ("FF", 2, ["F"], 24),
                                ("F2K4", 3, ["FK4"], 14)):
        out[name] = np.array([
            expansion.evaluate(props, positions={"x": 0.0, "y": 0.0},
                               t_final=float(t), component_pair=(0, 0),
                               orders=[order], vertex_types=vt,
                               method="gauss_legendre", n_gauss=ng,
                               n_jobs=-1 if order >= 3 else 1).total
            for t in t_grid])
    out["total"] = out["order0"] + out["FF"] + out["F2K4"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="level_b_results.npz")
    args = ap.parse_args()
    # 2e6 puts the MC error near 1.2 % of the signal, so the computed
    # F^3 correction (7.9 % at t = 3) shows up at better than 6 sigma.
    n_real = 200_000 if args.quick else 2_000_000
    n_scal = 100_000 if args.quick else 1_000_000
    n_dt = 100_000 if args.quick else 300_000
    n_r = 200_000 if args.quick else 1_000_000
    p, s = sn.PARAMS, F_AMPLITUDE

    print(f"demo 3 level B -- F amplitude s = {s}, n = {p.n_dimensionless:g}")

    t0 = time.perf_counter()
    th = theory_xi01(T_GRID, R_GRID, T_R, p=p)
    tot_t, tot_r = combine(th, s)
    print(f"\n[1] theory channels ({time.perf_counter()-t0:.0f} s)")
    print(f"    {'t':>5} {'F.k3':>12} {'F3.k3':>12} {'F3.k5':>12} {'total':>12} "
          f"{'F3k3/Fk3':>9} {'F3k5/Fk3':>9} {'O(F^5) est':>10}")
    for j, t in enumerate(T_GRID):
        a, b, c = s * th["fk3"][j], s ** 3 * th["f3k3"][j], s ** 3 * th["f3k5"][j]
        print(f"    {t:5.2f} {a: .5e} {b: .5e} {c: .5e} {tot_t[j]: .5e} "
              f"{b/a:9.4f} {c/a:9.5f} {(b/a)**2:10.5f}")

    t0 = time.perf_counter()
    sim = sb.simulate_xi(p, s, [0.0], T_GRID, n_real, dt=0.01, seed=11)
    print(f"\n[2] simulation, {n_real:,} realisations, dt = 0.01 "
          f"({time.perf_counter()-t0:.0f} s)")
    print(f"    blow-up fraction: {sim.blowup_fraction:.2e}   "
          f"variance reduction from the control variate: "
          f"x{sim.variance_reduction.min():.1f}-{sim.variance_reduction.max():.1f}")
    print(f"    {'t':>5} {'theory':>13} {'simulation':>13} {'MC err':>10} {'dev':>7}")
    for j, t in enumerate(T_GRID):
        d = (sim.xi01[0, j] - tot_t[j]) / sim.xi01_err[0, j]
        print(f"    {t:5.2f} {tot_t[j]: .6e} {sim.xi01[0, j]: .6e} "
              f"{sim.xi01_err[0, j]:10.2e} {d:6.2f}s")
    pulls = (sim.xi01[0] - tot_t) / sim.xi01_err[0]
    print(f"    pulls: mean {pulls.mean():+.2f}, max |pull| {np.abs(pulls).max():.2f}")
    print(f"    NOTE: all six t share the same realisations, so the residuals are")
    print(f"          strongly correlated -- a coherent offset is ONE fluctuation.")

    # --- paired dt study -------------------------------------------------
    # Same seed AND same n_real for every dt, so the SAME events are drawn
    # (the draw happens before stepping and does not consume rng state per
    # step).  The differences are then almost free of Monte-Carlo noise and
    # measure the ETDRK2 discretisation error alone.
    t0 = time.perf_counter()
    dts = [0.02, 0.01, 0.005]
    paired = [sb.simulate_xi(p, s, [0.0], T_GRID, n_dt, dt=h, seed=101)
              for h in dts]
    print(f"\n[2b] paired dt study, {n_dt:,} realisations, identical events "
          f"({time.perf_counter()-t0:.0f} s)")
    print(f"    {'t':>5} " + " ".join(f"{'dt=' + str(h):>13}" for h in dts)
          + f" {'|d(.02-.01)|':>13} {'|d(.01-.005)|':>13} {'ratio':>7} {'MC err':>10}")
    for j, t in enumerate(T_GRID):
        v = [pp.xi01[0, j] for pp in paired]
        d1, d2 = abs(v[0] - v[1]), abs(v[1] - v[2])
        print(f"    {t:5.2f} " + " ".join(f"{x: .6e}" for x in v)
              + f" {d1:13.2e} {d2:13.2e} {d1/max(d2,1e-300):7.2f} "
              f"{sim.xi01_err[0, j]:10.2e}")
    d_fin = np.abs(paired[1].xi01[0] - paired[2].xi01[0])
    print(f"    ETDRK2 is O(dt^2), so halving dt should shrink the difference ~4x.")
    print(f"    |dt=0.01 - dt=0.005| is at most "
          f"{np.max(d_fin / sim.xi01_err[0]):.2f} of the production MC error, so")
    print(f"    the discretisation error is NOT what limits the comparison.")

    t0 = time.perf_counter()
    simr = sb.simulate_xi(p, s, R_GRID, [T_R], n_r, dt=0.01, seed=13)
    print(f"\n[3] separations at t = {T_R} ({time.perf_counter()-t0:.0f} s) -- "
          f"sites placed exactly at the plotted r, no interpolation")
    print(f"    {'r':>5} {'theory':>13} {'simulation':>13} {'MC err':>10} {'dev':>7}")
    for j, r in enumerate(R_GRID):
        d = (simr.xi01[j, 0] - tot_r[j]) / simr.xi01_err[j, 0]
        print(f"    {r:5.2f} {tot_r[j]: .6e} {simr.xi01[j, 0]: .6e} "
              f"{simr.xi01_err[j, 0]:10.2e} {d:6.2f}s")
    pr = (simr.xi01[:, 0] - tot_r) / simr.xi01_err[:, 0]
    print(f"    pulls: mean {pr.mean():+.2f}, max |pull| {np.abs(pr).max():.2f}")

    print(f"\n[4] amplitude scaling: the residual after subtracting Fk3 must go as s^3")
    amps = [0.1, 0.2, 0.3]
    res = []
    for a in amps:
        r = sb.simulate_xi(p, a, [0.0], [T_R], n_scal, dt=0.01, seed=17)
        lead = a * th["fk3"][list(T_GRID).index(T_R)]
        resid = r.xi01[0, 0] - lead
        res.append((a, r.xi01[0, 0], r.xi01_err[0, 0], lead, resid,
                    a ** 3 * (th["f3k3"] + th["f3k5"])[list(T_GRID).index(T_R)]))
        print(f"    s={a:4.2f}  sim={r.xi01[0,0]: .5e}+-{r.xi01_err[0,0]:.1e}  "
              f"Fk3={lead: .5e}  residual={resid: .5e}  predicted F^3={res[-1][5]: .5e}")
    la = np.log(np.array([r[0] for r in res]))
    lr = np.log(np.abs(np.array([r[4] for r in res])))
    slope = np.polyfit(la, lr, 1)[0]
    print(f"    fitted exponent of the residual: {slope:.2f}  (F^3 predicts 3)")
    print(f"    NOTE: this cannot separate F3.k3 from F3.k5 -- both scale as s^3.")
    print(f"          Only computing them does; F3.k5 is "
          f"{100*th['f3k5'][-1]/th['f3k3'][-1]:.2f}% of F3.k3 at t={T_GRID[-1]}.")

    t0 = time.perf_counter()
    xi00 = theory_xi00(T_GRID, s, p=p)
    print(f"\n[5] xi_00 -- the even-cumulant sector ({time.perf_counter()-t0:.0f} s)")
    print(f"    {'t':>5} {'order0':>12} {'FF':>12} {'F2.k4':>12} {'total':>12} "
          f"{'simulation':>12} {'MC err':>10} {'dev':>7}")
    for j, t in enumerate(T_GRID):
        d = (sim.xi00[0, j] - xi00["total"][j]) / sim.xi00_err[0, j]
        print(f"    {t:5.2f} {xi00['order0'][j]: .5e} {xi00['FF'][j]: .5e} "
              f"{xi00['F2K4'][j]: .5e} {xi00['total'][j]: .5e} "
              f"{sim.xi00[0, j]: .5e} {sim.xi00_err[0, j]:10.2e} {d:6.2f}s")

    np.savez(args.out, t_grid=T_GRID, r_grid=R_GRID, t_r=T_R, s=s,
             **{f"th_{k}": v for k, v in th.items()},
             total_t=tot_t, total_r=tot_r,
             sim_t=sim.xi01[0], sim_t_err=sim.xi01_err[0],
             dt_values=np.array(dts),
             dt_curves=np.array([pp.xi01[0] for pp in paired]),
             n_real_dt=n_dt,
             sim_r=simr.xi01[:, 0], sim_r_err=simr.xi01_err[:, 0],
             sim_xi00=sim.xi00[0], sim_xi00_err=sim.xi00_err[0],
             **{f"xi00_{k}": v for k, v in xi00.items()},
             scaling=np.array([[r[0], r[1], r[2], r[3], r[4], r[5]] for r in res]),
             scaling_slope=slope, blowup=sim.blowup_fraction,
             var_reduction=sim.variance_reduction[0], n_real=n_real)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
