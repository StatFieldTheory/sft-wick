r"""Demo 8: run the four models against the exact reference.

Every row compares one package route with the moment hierarchy of
:mod:`time8_reference` (or, for the Gaussian kernel, with the hand
contraction), and is written to ``time8_results.json``::

    python time8_run.py            # 580 comparisons, ~25 min
    python time8_run.py --items ad # only the models named

Routes.  ``exact C`` gives the package the defining integrals of C
(:class:`time8_reference.ExactC`) as its ``c_closed_form``, so a row
measures the diagrams and R alone; ``table`` is the package's own C
quadrature and spline table.  A matrix R (component-dependent rates)
runs on the scalar loops only, so the rate-dependent models are run at
unequal rates there and at equal rates on Gauss-Legendre.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from dataclasses import replace

os.environ.setdefault("SFT_WICK_QUIET_CACHE", "1")

import numpy as np  # noqa: E402

import time8_params as pm  # noqa: E402
import time8_reference as ref  # noqa: E402
import time8_system as ts  # noqa: E402

POS = pm.POSITIONS
ONE = ("phi_a(x)",)
TWO = ("phi_a(x)", "phi_b(y)")
THREE = ("phi_a(x)", "phi_b(y)", "phi_c(z)")
ROWS: list[dict] = []


def _rel(got: float, want: float) -> float:
    return abs(got - want) / abs(want) if want else abs(got)


def record(item, route, obs, order, comps, got, want, seconds, note=""):
    row = dict(item=item, route=route,
               observable="<" + " ".join(f"phi_{c}" for c in comps) + ">",
               order=order, comps=list(comps), note=note,
               package=float(got), reference=float(want),
               rel=_rel(got, want), seconds=round(seconds, 2))
    ROWS.append(row)
    return row


def evaluate(exp, props, comps, order, method, t_final, **kw):
    t0 = time.perf_counter()
    val = exp.evaluate(props, positions=POS, t_final=t_final, component_pair=comps,
                       orders=[order], method=method, **kw).total
    return val, time.perf_counter() - t0


def run_plan(item, route, exp_by_obs, props, plan, refs, methods, t_final,
             **common):
    """One route over a plan of ``(observable, order, comps, note)``."""
    for method in methods:
        kw = dict(ts.METHOD_KW[method], **common)
        cells = []
        for obs, order, comps, note in plan:
            want = refs[(obs, order, comps, note)]
            if want == 0.0:
                continue
            try:
                got, secs = evaluate(exp_by_obs[(obs, order, note)], props, comps,
                                     order, method, t_final, **kw)
            except NotImplementedError as exc:
                print(f"  {route}, {method:15s}: refused -- {str(exc).splitlines()[0][:70]}")
                cells = None
                break
            row = record(item, f"{route}, {method}", obs, order, comps, got, want,
                         secs, note)
            cells.append(f"{''.join(map(str, comps))}{note}o{order} {row['rel']:.1e}")
        if cells:
            print(f"  {route}, {method:15s}: " + "  ".join(cells))


# ---------------------------------------------------------------------------
# (a) a decay rate varying in time
# ---------------------------------------------------------------------------

def item_a(quick: bool) -> None:
    print("\n(a) DiagonalA with a callable rate "
          "gamma(t) = [1 + 0.5 sin t, 0.6 + 0.3 cos 2t]")
    for t_min in (-1.3, 0.7):
        for rate, label, methods in (
                (pm.UNEQUAL_RATE, "unequal rates (matrix R)",
                 ["nquad", "qmc_scalar", "qmc", "gauss_legendre", "qmc_vectorized"]),
                (pm.EQUAL_RATE, "equal rates (scalar R)",
                 ["gauss_legendre", "qmc_vectorized", "nquad"])):
            p = replace(pm.RateParams(), rate=rate, t_min=t_min)
            system = ts.rate_system(p, t_max_cache=p.t_final + 1.0, n_grid_cache=800)
            cf = ts.exact_C(p, n=24, has_diagonal_kink=True)
            props = ts.exact_propagators(system, cf, t_max=p.t_final + 0.3)
            models = {n: ref.rate_model(p, [POS[k] for k in "xyz"[:n]])
                      for n in (1, 2, 3)}
            plan = [(ONE, 1, (0,), ""), (ONE, 1, (1,), "")]
            plan += [(TWO, o, c, "") for o in (0, 2)
                     for c in itertools.product(range(2), repeat=2)]
            plan += [(THREE, 1, c, "") for c in ((0, 1, 1), (1, 0, 0))]
            refs, exps = {}, {}
            for obs, order, comps, note in plan:
                legs = [("phi", c, i) for i, c in enumerate(comps)]
                refs[(obs, order, comps, note)] = models[len(obs)].moments(
                    legs, [(order,)], p.t_min, p.t_final)[(order,)]
                exps.setdefault((obs, order, note),
                                system.expand(obs, orders=[order]))
            print(f" t_min = {t_min}, {label}")
            run_plan("a", "exact C", exps, props, plan, refs,
                     methods if not quick else methods[:1], p.t_final)
            grids = [21] if quick else [21, 41]
            for n_grid_t in grids:
                tab = ts.table_propagators(system, p.t_final + 0.3, n_grid_t,
                                           c_method="gauss_legendre", c_n_gauss=24)
                run_plan("a", f"table n_grid_t={n_grid_t}", exps, tab, plan, refs,
                         ["nquad"] if not system.iso_R else ["gauss_legendre"],
                         p.t_final)
    item_a_rate_grid()


def item_a_rate_grid() -> None:
    """How the moments converge in the spacing of the rate-cache grid."""
    p = replace(pm.RateParams(), rate=pm.EQUAL_RATE)
    m1 = ref.rate_model(p, [POS["x"]])
    m2 = ref.rate_model(p, [POS["x"], POS["y"]])
    refs = {(ONE, 1, (0,), ""): m1.moments([("phi", 0, 0)], [(1,)], p.t_min,
                                           p.t_final)[(1,)]}
    for comps in ((0, 1), (1, 1)):
        refs[(TWO, 2, comps, "")] = m2.moments(
            [("phi", comps[0], 0), ("phi", comps[1], 1)], [(2,)], p.t_min,
            p.t_final)[(2,)]
    plan = list(refs)
    cf = ts.exact_C(p, n=24, has_diagonal_kink=True)
    print(" rate-cache grid (exact C, gauss_legendre)")
    for t_max_cache, n_grid_cache in ((100.0, None), (2.9, 200), (2.9, 800)):
        system = ts.rate_system(p, t_max_cache=t_max_cache,
                                n_grid_cache=n_grid_cache)
        lin = system._effective_linear
        h = (lin.t_max_cache - lin.t_min_cache) / (lin.n_grid_cache - 1)
        props = ts.exact_propagators(system, cf, t_max=p.t_final + 0.3)
        exps = {(obs, order, note): system.expand(obs, orders=[order])
                for obs, order, comps, note in plan}
        run_plan("a", f"rate grid h={h:.3f}", exps, props, plan, refs,
                 ["gauss_legendre"], p.t_final)


# ---------------------------------------------------------------------------
# (b) a damped oscillator
# ---------------------------------------------------------------------------

def item_b(quick: bool) -> None:
    p = pm.OscillatorParams()
    print(f"\n(b) ExplicitR, damped oscillator omega={p.R.omega} zeta={p.R.zeta} "
          f"(R changes sign at tau = {np.pi / p.R.omega_d:.2f}), white force")
    system = ts.oscillator_system(p)
    cf = ts.exact_C(p, n=24)
    props = ts.exact_propagators(system, cf, t_max=p.t_final + 0.3)
    models = {n: ref.oscillator_model(p, [POS[k] for k in "xyz"[:n]])
              for n in (1, 2, 3)}
    #: (observable, order, components, vertex-type note) and the tag of the
    #: reference: F counts the quadratic vertex, G the cubic one.
    plan_tags = [
        (ONE, 1, (0,), "F", (1, 0)), (ONE, 1, (1,), "F", (1, 0)),
        (ONE, 2, (0,), "FG", (1, 1)), (ONE, 2, (1,), "FG", (1, 1)),
        (TWO, 0, (0, 1), "", (0, 0)), (TWO, 0, (1, 1), "", (0, 0)),
        (TWO, 1, (0, 1), "G", (0, 1)), (TWO, 1, (1, 1), "G", (0, 1)),
        (TWO, 2, (0, 1), "F", (2, 0)), (TWO, 2, (1, 1), "F", (2, 0)),
        (TWO, 2, (0, 1), "G", (0, 2)), (TWO, 2, (1, 1), "G", (0, 2)),
        (THREE, 1, (0, 1, 1), "F", (1, 0)), (THREE, 1, (1, 0, 0), "F", (1, 0)),
    ]
    plan, refs, exps = [], {}, {}
    for obs, order, comps, note, tag in plan_tags:
        legs = [("x", c, i) for i, c in enumerate(comps)]
        refs[(obs, order, comps, note)] = models[len(obs)].moments(
            legs, [tag], p.t_min, p.t_final)[tag]
        exps[(obs, order, note)] = system.expand(obs, orders=[order])
        plan.append((obs, order, comps, note))
    for note in {n for _, _, _, n, _ in plan_tags}:
        sub = [row for row in plan if row[3] == note]
        vt = None if note == "" else {note}
        methods = ["gauss_legendre", "qmc_vectorized", "nquad", "qmc_scalar"]
        run_plan("b", "exact C", exps, props, sub, refs,
                 methods if not quick else methods[:1], p.t_final,
                 vertex_types=vt)
    for n_grid_t in ([21] if quick else [21, 41]):
        tab = ts.table_propagators(system, p.t_final + 0.3, n_grid_t,
                                   c_method="gauss_legendre", c_n_gauss=20)
        for note in {n for _, _, _, n, _ in plan_tags}:
            sub = [row for row in plan if row[3] == note]
            run_plan("b", f"table n_grid_t={n_grid_t}", exps, tab, sub, refs,
                     ["gauss_legendre"], p.t_final,
                     vertex_types=None if note == "" else {note})
    item_b_two_times(p, system, props, models[2], cf, quick)


def item_b_two_times(p, system, props, model, cf, quick: bool) -> None:
    """Unequal external times.  C is kinked where an internal time crosses
    the earlier external time; the integrators cut the internal variable's
    range there (README, "Limits"), which is what keeps Gauss-Legendre
    exponential.  Before the cut this channel had no rate at all (4.5e-04 to
    2.2e-07 between 8 and 48 nodes)."""
    T1, T2 = p.t_final - 0.9, p.t_final
    print(f" unequal external times x: {T1:.2f}, y: {T2:.2f}")
    cases = [(0, "", (0, 0)), (1, "G", (0, 1)), (2, "F", (2, 0)), (2, "G", (0, 2))]
    for order, note, tag in cases:
        exp = system.expand(TWO, orders=[order])
        for comps in ((0, 1), (1, 1)):
            want = model.two_time([("x", comps[0], 0)], T1, [("x", comps[1], 1)],
                                  T2, [tag], p.t_min)[tag]
            cells = []
            for method, kw in (("qmc_vectorized", dict(n_samples=2 ** 16, seed=3)),
                               ("nquad", {}),
                               ("gauss_legendre", dict(n_gauss=12)),
                               ("gauss_legendre", dict(n_gauss=28))):
                if quick and method != "qmc_vectorized":
                    continue
                got, secs = evaluate(exp, props, comps, order, method, T2,
                                     vertex_types=None if note == "" else {note},
                                     external_times={"x": T1, "y": T2}, **kw)
                tag_name = method + (f" n_gauss={kw['n_gauss']}"
                                     if method == "gauss_legendre" else "")
                row = record("b", f"two times, {tag_name}", TWO, order, comps,
                             got, want, secs, note)
                cells.append(f"{tag_name} {row['rel']:.1e}")
            print(f"  {''.join(map(str, comps))}{note}o{order}: " + "  ".join(cells))
    item_b_two_times_by_hand(p, model, cf, T1, T2)


def item_b_two_times_by_hand(p, model, cf, T1: float, T2: float) -> None:
    """The order-1 cubic channel by hand, with and without the split at the
    kink: the diagnosis the package's cut is based on -- the same integral
    converges when its domain is split at the crossing and not otherwise,
    whoever writes it out."""
    r = abs(POS["x"] - POS["y"])

    def C(b, t1, t2, rr):
        t1, t2, rr = np.broadcast_arrays(np.asarray(t1, float),
                                         np.asarray(t2, float),
                                         np.asarray(rr, float))
        return cf.diagonal(t1.ravel(), t2.ravel(),
                           rr.ravel())[:, b].reshape(t1.shape)

    hand = ref.HandContraction(
        lambda a, t, s: p.R.tau(np.asarray(t) - np.asarray(s)), C, p.t_min)
    for comps in ((0, 1), (1, 1)):
        want = model.two_time([("x", comps[0], 0)], T1, [("x", comps[1], 1)],
                              T2, [(0, 1)], p.t_min)[(0, 1)]
        cells = []
        for split, nodes in ((True, 60), (True, 120), (False, 60), (False, 240)):
            got = hand.two_point_order1_cubic(comps[0], comps[1], T1, T2, r,
                                              pm.G_CUBIC, split=split, n=nodes)
            row = record("b", f"two times by hand, split={split} n={nodes}",
                         TWO, 1, comps, got, want, 0.0, "G")
            cells.append(f"{'split' if split else 'one piece'} n={nodes} "
                         f"{row['rel']:.1e}")
        print(f"  {''.join(map(str, comps))}Go1 by hand: " + "  ".join(cells))


# ---------------------------------------------------------------------------
# (c) custom temporal kernels
# ---------------------------------------------------------------------------

def item_c(quick: bool) -> None:
    print("\n(c) custom temporal kernels through CustomKernel")
    for kern in (pm.Matern32(), pm.DampedCosine()):
        for gamma, label, methods in (
                ((1.1, 1.1), "equal rates (scalar R)",
                 ["gauss_legendre", "qmc_vectorized"]),
                ((0.8, 1.3), "unequal rates (matrix R)",
                 ["nquad", "qmc_scalar", "gauss_legendre"])):
            p = pm.KernelParams(kernel=kern, gamma=gamma)
            system = ts.kernel_system(p)
            cf = ts.exact_C(p, n=32, has_diagonal_kink=True)
            props = ts.exact_propagators(system, cf, t_max=p.t_final + 0.3)
            models = {n: ref.kernel_model(p, [POS[k] for k in "xyz"[:n]])
                      for n in (1, 2)}
            plan = [(ONE, 1, (0,), ""), (ONE, 1, (1,), "")]
            plan += [(TWO, o, c, "") for o in (0, 2)
                     for c in ((0, 1), (1, 1), (0, 0))]
            refs, exps = {}, {}
            for obs, order, comps, note in plan:
                legs = [("phi", c, i) for i, c in enumerate(comps)]
                refs[(obs, order, comps, note)] = models[len(obs)].moments(
                    legs, [(order,)], p.t_min, p.t_final)[(order,)]
                exps.setdefault((obs, order, note),
                                system.expand(obs, orders=[order]))
            print(f" {type(kern).__name__}, {label}")
            run_plan("c", f"{type(kern).__name__} exact C", exps, props, plan, refs,
                     methods if not quick else methods[:1], p.t_final)
            for n_grid_t in ([21] if quick else [21, 41]):
                tab = ts.table_propagators(system, p.t_final + 0.3, n_grid_t,
                                           c_method="gauss_legendre", c_n_gauss=24)
                run_plan("c", f"{type(kern).__name__} table n_grid_t={n_grid_t}",
                         exps, tab, plan, refs,
                         ["gauss_legendre"] if system.iso_R else ["nquad"],
                         p.t_final)
    item_c_gaussian(quick)


def item_c_gaussian(quick: bool) -> None:
    """GaussianTemporal: no finite embedding.  The reference is the hand
    contraction of the order-1 and order-2 diagrams with C in closed form."""
    print(" GaussianTemporal (no embedding: hand contraction with the erf C)")
    for gamma, label, methods in (((1.1, 1.1), "equal rates",
                                   ["gauss_legendre", "qmc_vectorized"]),
                                  ((0.8, 1.3), "unequal rates",
                                   ["nquad", "qmc_scalar"])):
        p = pm.KernelParams(kernel=pm.GaussianKernel(), gamma=gamma)
        env = pm.GaussianEnvelope(p.sigma_x)

        def C(b, t1, t2, r, p=p, env=env):
            return env(r) * ref.gaussian_kernel_C(
                p.gamma[b], p.kernel.lam, p.kernel.sigma,
                np.asarray(t1, float) - p.t_min, np.asarray(t2, float) - p.t_min)

        def R(a, t, s, p=p):
            return np.exp(-p.gamma[a] * (np.asarray(t) - np.asarray(s)))

        hand = ref.HandContraction(R, C, p.t_min, n=48)
        r = abs(POS["x"] - POS["y"])
        system = ts.kernel_system(p)
        props = ts.exact_propagators(system, ts.exact_C(p, n=40),
                                     t_max=p.t_final + 0.3)
        plan = [(ONE, 1, (0,), ""), (ONE, 1, (1,), "")]
        plan += [(TWO, o, c, "") for o in (0, 2) for c in ((0, 1), (1, 1))]
        refs, exps = {}, {}
        for obs, order, comps, note in plan:
            if obs is ONE:
                want = hand.tadpole(comps[0], p.t_final)
            elif order == 0:
                want = hand.two_point_order0(comps[0], comps[1], p.t_final, r)
            else:
                want = hand.two_point_order2(comps[0], comps[1], p.t_final, r)
            refs[(obs, order, comps, note)] = want
            exps.setdefault((obs, order, note), system.expand(obs, orders=[order]))
        print(f"  {label}")
        run_plan("c", "Gaussian exact C", exps, props, plan, refs,
                 methods if not quick else methods[:1], p.t_final)
        for n_grid_t in ([21] if quick else [21, 41]):
            tab = ts.table_propagators(system, p.t_final + 0.3, n_grid_t)
            run_plan("c", f"Gaussian table n_grid_t={n_grid_t}", exps, tab, plan,
                     refs, ["gauss_legendre"] if system.iso_R else ["nquad"],
                     p.t_final)
    item_c_translation()


def item_c_translation() -> None:
    """The kernel is stationary and the rates constant, so shifting t_min and
    the observation time together changes nothing."""
    print(" time-translation invariance (GaussianTemporal, table C)")
    base = None
    for shift in (-1.1, 0.0, 0.9):
        p = pm.KernelParams(kernel=pm.GaussianKernel(), gamma=(1.1, 1.1),
                            t_min=shift)
        system = ts.kernel_system(p)
        tab = ts.table_propagators(system, p.t_final + 0.3, 31)
        exp = system.expand(TWO, orders=[0, 2])
        vals = [exp.evaluate(tab, positions=POS, t_final=p.t_final,
                             component_pair=(1, 1), orders=[o],
                             method="gauss_legendre", n_gauss=12).total
                for o in (0, 2)]
        base = vals if base is None else base
        for o, v, b in zip((0, 2), vals, base):
            record("c", "time translation", TWO, o, (1, 1), v, b, 0.0,
                   note=f"t_min={shift}")
        print(f"  t_min={shift:+.1f}: order 0 {vals[0]:+.12e} rel "
              f"{_rel(vals[0], base[0]):.1e}   order 2 {vals[1]:+.12e} rel "
              f"{_rel(vals[1], base[1]):.1e}")


# ---------------------------------------------------------------------------
# (d) a white-noise amplitude varying in time
# ---------------------------------------------------------------------------

def item_d(quick: bool) -> None:
    print("\n(d) CustomImpulse with s_a(t), and the white-noise C table")
    for gamma, label, methods in (((1.2, 1.2), "equal rates (scalar R)",
                                   ["gauss_legendre", "qmc_vectorized", "nquad"]),
                                  ((0.9, 1.4), "unequal rates (matrix R)",
                                   ["nquad", "qmc_scalar"])):
        p = pm.WhiteParams(gamma=gamma)
        system = ts.white_system(p)
        cf = ts.exact_C(p, n=32)
        props = ts.exact_propagators(system, cf, t_max=p.t_final + 0.3)
        models = {n: ref.white_model(p, [POS[k] for k in "xyz"[:n]])
                  for n in (1, 2)}
        plan = [(ONE, 1, (0,), ""), (ONE, 1, (1,), "")]
        plan += [(TWO, o, c, "") for o in (0, 2) for c in ((0, 1), (1, 1))]
        refs, exps = {}, {}
        for obs, order, comps, note in plan:
            legs = [("phi", c, i) for i, c in enumerate(comps)]
            refs[(obs, order, comps, note)] = models[len(obs)].moments(
                legs, [(order,)], p.t_min, p.t_final)[(order,)]
            exps.setdefault((obs, order, note), system.expand(obs, orders=[order]))
        print(f" {label}")
        run_plan("d", "exact C", exps, props, plan, refs,
                 methods if not quick else methods[:1], p.t_final)
        # The step sweep runs on the scalar-R variant: at n_grid_t = 81 the
        # table's spline oscillates around the white-noise kink finely enough
        # that nquad, the only integrator a matrix R has left, subdivides for
        # tens of minutes on one order-2 diagram.
        grids = ([21] if quick else
                 [11, 21, 41, 81] if system.iso_R else [11, 21, 41])
        for n_grid_t in grids:
            t0 = time.perf_counter()
            tab = ts.table_propagators(system, p.t_final + 0.3, n_grid_t,
                                       c_method="gauss_legendre", c_n_gauss=24)
            build = time.perf_counter() - t0
            run_plan("d", f"table n_grid_t={n_grid_t}", exps, tab, plan, refs,
                     ["gauss_legendre", "qmc_vectorized"] if system.iso_R
                     else ["nquad"], p.t_final)
            print(f"   (table build {build:.1f} s)")
    item_d_table_pointwise()


def item_d_table_pointwise() -> None:
    """Where the table's C is wrong: the kink on its time diagonal."""
    p = pm.WhiteParams(gamma=(1.2, 1.2))
    system = ts.white_system(p)
    cf = ts.exact_C(p, n=32)
    print(" C table vs the exact C, by distance from the time diagonal")
    for n_grid_t in (21, 41, 81):
        tab = ts.table_propagators(system, p.t_final + 0.3, n_grid_t,
                                   c_method="gauss_legendre", c_n_gauss=24)
        t1 = np.linspace(p.t_min + 0.2, p.t_final, 37)
        r = abs(POS["x"] - POS["y"])
        cells = []
        for delta in (0.0, 1e-6, 1e-2, 0.1, 0.4):
            t2 = t1 - delta
            got = tab.cache.C_at_batch(t1, t2, np.zeros_like(t1),
                                       np.full_like(t1, r))
            want = cf.diagonal(t1, t2, np.full_like(t1, r))
            err = float(np.max(np.abs(got - want)) / np.abs(want).max())
            record("d", f"C table n_grid_t={n_grid_t}", TWO, 0, (0, 0),
                   0.0, 0.0, 0.0, note=f"delta={delta}")
            ROWS[-1]["rel"] = err
            cells.append(f"delta={delta:<6g} {err:.1e}")
        print(f"  n_grid_t={n_grid_t:3d}: " + "  ".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="time8_results.json")
    ap.add_argument("--items", default="abcd")
    ap.add_argument("--quick", action="store_true",
                    help="one integrator and one grid per route")
    args = ap.parse_args()
    t0 = time.perf_counter()
    for name in args.items:
        {"a": item_a, "b": item_b, "c": item_c, "d": item_d}[name](args.quick)
    worst: dict = {}
    for row in ROWS:
        key = f"{row['item']}: {row['route']}"
        worst[key] = max(worst.get(key, 0.0), row["rel"])
    with open(args.out, "w") as fh:
        json.dump(dict(rows=ROWS, worst=worst,
                       seconds=round(time.perf_counter() - t0, 1)), fh, indent=1)
    print(f"\n{len(ROWS)} comparisons, {time.perf_counter() - t0:.0f} s; "
          f"wrote {args.out}")


if __name__ == "__main__":
    main()
