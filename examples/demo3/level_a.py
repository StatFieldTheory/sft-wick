r"""Level A --- the exact test (``F = 0``).

With no interaction the perturbative series *terminates*: the ``m``-point
connected function is a single diagram, the ``m`` external ``φ``\ s
contracted with the ``m`` ``ψ``\ s of the ``κ^(m)`` vertex, so

    ``⟨φ(z'_1) … φ(z'_m)⟩_c = K_R(z'_1, …, z'_m)``   exactly.

No truncation, no interacting correction, and --- unlike level B --- **no
neglected cumulant ladder**: a ``κ^(m')`` vertex with ``m' ≠ m`` cannot
balance the legs, so nothing else can mix in.  Three mutually independent
routes to the same number:

1. the closed form ``ν h^m X_m T̃_m``;
2. the package, through the full diagram machinery;
3. an **event-exact** simulation --- no time stepping and no spatial
   discretisation, so the only error is Monte Carlo.

Run ``python level_a.py`` (add ``--quick`` for a fast pass).  Results are
cached to ``level_a_results.npz`` for ``make_figures.py``.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

os.environ.setdefault("SFT_WICK_QUIET_CACHE", "1")

import shot_noise as sn
import simulate as sim
import system as dsys

T_GRID = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
R_GRID = np.array([0.0, 0.25, 0.5, 1.0, 1.5, 2.0])
N_SWEEP = np.array([0.25, 1.0, 4.0])
T_NSWEEP = 1.5


# =====================================================================
# Theory: closed form and package
# =====================================================================

def closed_form_3pt(t_grid, positions=(0.0, 0.0, 0.0), p=sn.PARAMS):
    xs = np.repeat(np.array(positions, float)[:, None], len(t_grid), axis=1)
    ts = np.repeat(np.atleast_2d(t_grid), 3, axis=0)
    return sn.K_R(xs, ts, p)


def closed_form_4pt(t_grid, p=sn.PARAMS):
    xs = np.zeros((4, len(t_grid)))
    ts = np.repeat(np.atleast_2d(t_grid), 4, axis=0)
    return sn.K_R(xs, ts, p)


def package_npt(t_grid, m, positions=None, p=sn.PARAMS, n_gauss=20):
    """Evaluate the level-A ``m``-point function through the package.

    Every external operator gets its **own** spatial label even when the
    positions coincide: repeating a label silently loses pairing
    multiplicity on the base commit (see
    ``tests/test_demo3_levels.py::test_coincident_spatial_labels_agree_with_distinct_ones``).
    """
    labels = ["w", "x", "y", "z"][-m:]
    obs = tuple(f"phi_{c}({lab})" for c, lab in zip("abcd", labels))
    positions = (0.0,) * m if positions is None else positions
    system = dsys.make_system(p, cumulants=(m,))
    props = system.propagators(t_max=float(max(t_grid)) * 1.05 + 1.0,
                               n_grid_t=40, c_closed_form="auto",
                               c_closed_form_only=True, progress=False)
    expansion = system.expand(obs, orders=[1])
    out = []
    for t in t_grid:
        res = expansion.evaluate(
            props, positions=dict(zip(labels, positions)), t_final=float(t),
            component_pair=(0,) * m, orders=[1],
            method="gauss_legendre", n_gauss=n_gauss)
        out.append(res.total)
    return np.array(out)


# =====================================================================
# Simulation
# =====================================================================

def simulate_moments(t_grid, n_real, p=sn.PARAMS, seed=0, batch=20_000):
    """Event-exact ``⟨φ³⟩`` and connected ``⟨φ⁴⟩`` on the time grid.

    One event draw per realisation serves every ``t``: the same
    trajectory is observed at all times, which is both cheaper and
    exactly what the theory curve describes.
    """
    rng = np.random.default_rng(seed)
    window = sim.Window.for_times(float(max(t_grid)), p)
    phi = sim.sample_points(rng, np.zeros_like(t_grid), t_grid, n_real,
                            window, p, kind="phi", batch=batch)[:, 0, :]
    m3, e3, m4, e4 = [], [], [], []
    for j in range(len(t_grid)):
        col = phi[:, j: j + 1]
        est, se = sim.central_moments(np.repeat(col, 3, axis=1), 3)
        m3.append(est), e3.append(se)
        est4, se4 = sim.connected_cumulant(np.repeat(col, 4, axis=1))
        m4.append(est4), e4.append(se4)
    return (np.array(m3), np.array(e3), np.array(m4), np.array(e4),
            sim.truncation_bound(window, 3, p))


def simulate_separations(r_grid, t_final, n_real, p=sn.PARAMS, seed=1,
                         batch=20_000):
    """``⟨φ(0,t) φ(0,t) φ(r,t)⟩`` --- sites placed *exactly* at the plotted
    separations, so there is no interpolation anywhere."""
    rng = np.random.default_rng(seed)
    window = sim.Window.for_times(t_final, p)
    xs = np.concatenate([[0.0], r_grid])
    ts = np.full(len(xs), t_final)
    phi = sim.sample_points(rng, xs, ts, n_real, window, p, kind="phi",
                            batch=batch)[:, 0, :]
    est, err = [], []
    for j in range(len(r_grid)):
        trip = np.stack([phi[:, 0], phi[:, 0], phi[:, j + 1]], axis=1)
        e, s = sim.central_moments(trip, 3)
        est.append(e), err.append(s)
    return np.array(est), np.array(err)


def simulate_n_sweep(n_values, t_final, n_real, p=sn.PARAMS, seed=2,
                     batch=20_000):
    """``⟨φ³⟩`` at fixed ``t`` across the non-Gaussianity knob ``n``.

    ``κ₂`` is held fixed by compensating ``h``, so the Gaussian sector is
    identical at every ``n`` and only the non-Gaussian channel moves ---
    as ``1/√n``.
    """
    rows = []
    for n in n_values:
        q = p.with_n(float(n))
        rng = np.random.default_rng(seed + int(1000 * n))
        window = sim.Window.for_times(t_final, q)
        phi = sim.sample_points(rng, [0.0], [t_final], n_real, window, q,
                                kind="phi", batch=batch)[:, 0, :]
        est, se = sim.central_moments(np.repeat(phi, 3, axis=1), 3)
        exact = float(sn.K_R(np.zeros((3, 1)), np.full((3, 1), t_final), q)[0])
        rows.append((n, est, se, exact, q.skewness, q.variance))
    return rows


# =====================================================================
# Driver
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="small realisation counts for a fast pass")
    ap.add_argument("--out", default="level_a_results.npz")
    args = ap.parse_args()
    # One event draw serves both moments, so the count is set by the
    # hungrier of the two -- the connected 4-point, whose estimator
    # subtracts three pair products and so has a much larger variance.
    n_moments = 60_000 if args.quick else 1_200_000
    nr = 60_000 if args.quick else 400_000
    p = sn.PARAMS

    print(f"demo 3 level A -- F = 0, n = {p.n_dimensionless:g}, "
          f"skewness {p.skewness:.4f}, kappa2 {p.variance:g}")
    print(f"  gamma = {p.gamma:g}, sigma_t = {p.sigma_t:g} "
          f"(1/sigma_t = {p.a:g}; no scale separation, gamma != 1/sigma_t)")

    t0 = time.perf_counter()
    exact3 = closed_form_3pt(T_GRID, p=p)
    exact4 = closed_form_4pt(T_GRID, p=p)
    pkg3 = package_npt(T_GRID, 3, p=p)
    pkg4 = package_npt(T_GRID, 4, p=p)
    t_theory = time.perf_counter() - t0

    print("\n[1] package vs closed form (the exact test)")
    r3 = np.abs(pkg3 - exact3) / np.abs(exact3)
    r4 = np.abs(pkg4 - exact4) / np.abs(exact4)
    print(f"    3-point: max relative difference {r3.max():.2e}")
    print(f"    4-point: max relative difference {r4.max():.2e}   ({t_theory:.1f} s)")

    t0 = time.perf_counter()
    sim3, err3, sim4, err4, trunc = simulate_moments(T_GRID, n_moments, p=p)
    t_sim = time.perf_counter() - t0
    print(f"\n[2] event-exact simulation, {n_moments:,} realisations ({t_sim:.0f} s)")
    print(f"    window truncation bound (m=3): {trunc['total']:.1e} relative")
    print(f"    {'t':>6} {'closed form':>13} {'simulation':>13} {'MC err':>10} {'dev':>7}")
    for j, t in enumerate(T_GRID):
        dev = (sim3[j] - exact3[j]) / err3[j]
        print(f"    {t:6.2f} {exact3[j]:13.6e} {sim3[j]:13.6e} {err3[j]:10.2e} {dev:6.2f}s")
    pull3 = (sim3 - exact3) / err3
    print(f"    3-point pulls: mean {pull3.mean():+.2f}, max |pull| {np.abs(pull3).max():.2f}")

    pull4 = (sim4 - exact4) / err4
    print(f"\n[3] connected 4-point, same {n_moments:,} realisations")
    print(f"    4-point pulls: mean {pull4.mean():+.2f}, max |pull| {np.abs(pull4).max():.2f}")
    print(f"    typical MC error {np.median(err4 / np.abs(exact4)) * 100:.1f}% of signal")

    t0 = time.perf_counter()
    simr, errr = simulate_separations(R_GRID, 2.0, nr, p=p)
    exactr = np.array([float(sn.K_R(np.array([[0.0], [0.0], [r]]),
                                    np.full((3, 1), 2.0), p)[0]) for r in R_GRID])
    pkgr = np.array([package_npt([2.0], 3, positions=(0.0, 0.0, float(r)), p=p)[0]
                     for r in R_GRID])
    print(f"\n[4] separations at t = 2 ({time.perf_counter()-t0:.0f} s)  "
          f"sites placed exactly at the plotted r -- no interpolation")
    pullr = (simr - exactr) / errr
    print(f"    package vs closed form: max rel {np.abs(pkgr-exactr).max()/np.abs(exactr).max():.2e}")
    print(f"    simulation pulls: max |pull| {np.abs(pullr).max():.2f}")

    print(f"\n[5] the 1/sqrt(n) law at t = {T_NSWEEP}")
    rows = simulate_n_sweep(N_SWEEP, T_NSWEEP, nr, p=p)
    print(f"    {'n':>6} {'skewness':>9} {'closed form':>13} {'simulation':>13} "
          f"{'MC err':>10} {'dev':>7} {'kappa2':>8}")
    for n, est, se, exact, skew, var in rows:
        print(f"    {n:6.2f} {skew:9.4f} {exact:13.6e} {est:13.6e} {se:10.2e} "
              f"{(est-exact)/se:6.2f}s {var:8.4f}")
    ratio = rows[0][3] / rows[-1][3]
    print(f"    closed form ratio n=0.25 / n=4 : {ratio:.4f}  "
          f"(1/sqrt(n) predicts {np.sqrt(4.0/0.25):.4f})")

    np.savez(args.out, t_grid=T_GRID, exact3=exact3, exact4=exact4,
             pkg3=pkg3, pkg4=pkg4, sim3=sim3, err3=err3, sim4=sim4,
             err4=err4, r_grid=R_GRID, exactr=exactr, pkgr=pkgr,
             simr=simr, errr=errr, n_sweep=np.array([r[0] for r in rows]),
             n_exact=np.array([r[3] for r in rows]),
             n_sim=np.array([r[1] for r in rows]),
             n_err=np.array([r[2] for r in rows]),
             n_skew=np.array([r[4] for r in rows]),
             n_real_moments=n_moments, n_real_r=nr,
             trunc_bound=trunc["total"])
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
