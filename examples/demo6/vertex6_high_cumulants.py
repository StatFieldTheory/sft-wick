r"""Demo 6, part D: ``m = 4`` and ``m = 5`` non-local vertices on every route.

Level A (``F = 0``): the connected ``m``-point function is the single
order-1 ``κ^(m)`` diagram, so the package's value is the R-contracted
cumulant exactly.  Four routes carry the same cumulant:

===================  =====================================================
route                what the package is given
===================  =====================================================
R-contracted         a callable returning ``K_R`` at the partner points
                     (``already_R_contracted=True``; the diagram has no
                     time integration left)
raw                  a callable returning the bare ``κ^(m)`` at the legs;
                     the package does the ``m`` leg integrals
                     (``equal_time`` for white pulses, where the cumulant
                     carries ``δ(t_i − t_j)``)
static               an ndarray: the cumulant of a force constant in space
                     and time
static ``equal_time``  an ndarray with ``δ(t_i − t_j)``: white jumps whose
                     jump vector does not depend on the point
===================  =====================================================

The first two use demo 4's compound-Poisson noise (``examples/demo4``),
whose cumulants Campbell's theorem gives in closed form; the last two use
demo 6's static force and white jumps, whose closed form is
:func:`vertex6_model.split_sum`.  Both are also checked against a moment
hierarchy (demo 4's for the pulses, demo 6's for the static vertices), at
equal and at distinct external times.

Run ``python vertex6_high_cumulants.py``; results go to
``high_cumulants_results.json``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("SFT_WICK_QUIET_CACHE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo4"))

import poisson_noise as nz  # noqa: E402
import poisson_system as dsys  # noqa: E402
from poisson_reference import Hierarchy as PoissonHierarchy  # noqa: E402
import vertex6_model as md  # noqa: E402
import vertex6_system as vs  # noqa: E402
from vertex6_reference import (Hierarchy, solve_multitime,  # noqa: E402
                               tag)

LABELS = ("x", "y", "z", "w", "v")
POS = {"x": 0.0, "y": 0.8, "z": -0.5, "w": 1.3, "v": 0.4}
T = 1.7
UNEQUAL = {"x": 1.7, "y": 1.2, "z": 0.6, "w": 1.45, "v": 0.95}
#: demo 6's own times start above its ``t_min = 0.4``.
STATIC_TIMES = {k: v + 0.3 for k, v in UNEQUAL.items()}


def rel(got, ref):
    return abs(got - ref) / abs(ref)


def _observable(m):
    labels = LABELS[:m]
    return labels, tuple(f"phi_{c}({lab})" for c, lab in zip("abcde", labels))


def poisson_routes(m, pp, rows):
    """demo 4's noise: the R-contracted and the raw callable."""
    labels, obs = _observable(m)
    xs = np.array([POS[lab] for lab in labels])
    tuples = list(itertools.product(range(2), repeat=m))
    for rc in (True, False):
        system = dsys.make_system(pp, cumulants=(m,), r_contracted=rc)
        props = dsys.propagators_for(system, pp, t_max=T + 0.5)
        exp = system.expand(obs, orders=[1], diag_C=False)
        if rc:
            routes = [("gauss_legendre", dict(n_gauss=4), tuples)]
        elif pp.pulse == "white":      # equal_time: one smooth time integral
            routes = [("gauss_legendre", dict(n_gauss=16), tuples)]
        else:
            # Kinked where two leg times cross, and sharply peaked for the
            # fast component (τ = 0.3).  Gauss-Legendre converges
            # algebraically and costs n^m callable evaluations per leg
            # order (m! of them), so it runs at m = 4 only.
            routes = [("qmc_vectorized", dict(n_samples=2 ** 18, seed=1),
                       tuples[:4])]
            if m == 4:
                routes.append(("gauss_legendre", dict(n_gauss=16),
                               tuples[:2]))
        for method, kw, tt in routes:
            t0 = time.perf_counter()
            for comps in tt:
                ref = float(nz.K_R(comps, xs, np.full(m, T), pp)[0])
                got = exp.evaluate(props, positions=POS, t_final=T,
                                   component_pair=comps, orders=[1],
                                   method=method, **kw).total
                rows.append(dict(noise=pp.pulse, m=m,
                                 route="R-contracted" if rc else "raw",
                                 method=method, times="equal",
                                 comps=list(comps), package=got,
                                 reference=ref, rel=rel(got, ref)))
            print(f"  {pp.pulse:12s} m={m} {'rc' if rc else 'raw':3s} "
                  f"{method:15s} {len(tt):2d} tuples "
                  f"({time.perf_counter() - t0:.1f} s)")
        if rc:                          # unequal external times
            ts = np.array([UNEQUAL[lab] for lab in labels])
            ext = {lab: UNEQUAL[lab] for lab in labels}
            for comps in tuples:
                ref = float(nz.K_R(comps, xs, ts, pp)[0])
                got = exp.evaluate(props, positions=POS,
                                   t_final=max(ext.values()),
                                   component_pair=comps, orders=[1],
                                   external_times=ext,
                                   method="gauss_legendre", n_gauss=4).total
                rows.append(dict(noise=pp.pulse, m=m, route="R-contracted",
                                 method="gauss_legendre", times="distinct",
                                 comps=list(comps), package=got,
                                 reference=ref, rel=rel(got, ref)))


def poisson_hierarchy(m, pp, rows):
    """demo 4's moment hierarchy against Campbell's closed form, at equal
    and at distinct times (the latter through :func:`solve_multitime`)."""
    labels, _obs = _observable(m)
    xs = np.array([POS[lab] for lab in labels])
    H = PoissonHierarchy(pp, list(xs))
    tg = (0, m - 2)
    for comps in [(0, 1, 1, 0, 1)[:m], (1, 1, 0, 1, 0)[:m]]:
        legs = [(c, i) for i, c in enumerate(comps)]
        ref = float(nz.K_R(comps, xs, np.full(m, T), pp)[0])
        got = H.moments(legs, [tg], T)[tg]
        rows.append(dict(noise=pp.pulse, m=m, route="hierarchy",
                         method="expm", times="equal", comps=list(comps),
                         package=got, reference=ref, rel=rel(got, ref)))
        ts = np.array([UNEQUAL[lab] for lab in labels])
        ref = float(nz.K_R(comps, xs, ts, pp)[0])
        mono = H.monomial(legs)
        freeze = {k: UNEQUAL[labels[i]] for (_a, i), k in H._phi.items()}
        out = solve_multitime(H.sde, [(mono, tg)], freeze, H._initial)
        rows.append(dict(noise=pp.pulse, m=m, route="hierarchy",
                         method="expm", times="distinct", comps=list(comps),
                         package=out[(mono, tg)], reference=ref,
                         rel=rel(out[(mono, tg)], ref)))


def static_routes(m, p, rows):
    """demo 6's static force and white jumps at distinct times."""
    labels, obs = _observable(m)
    ts = [STATIC_TIMES[lab] for lab in labels]
    ext = dict(zip(labels, ts))
    tuples = list(itertools.product(range(2), repeat=m))
    for kind, name in (("static", f"X{m}"), ("equal_time",
                                             f"J{m}" if m > 2 else "W2")):
        system = vs.make_system(p, static=(m,) if kind == "static" else (),
                                equal_time=(m,) if kind == "equal_time"
                                else ())
        props = vs.propagators_for(system, p, t_max=max(ts) + 0.5)
        exp = system.expand(obs, orders=[1], diag_C=False)
        tensors = (md.X_CUMULANTS if kind == "static" else md.JUMP_CUMULANTS)
        H = Hierarchy(
            p, [POS[lab] for lab in labels],
            sigma2=lambda a, x, b, y: md.sigma2(a, x, b, y, p),
            **({"x_cumulants": {m: tensors[m]}} if kind == "static"
               else {"jump_cumulants": {m: tensors[m]}}))
        tg = tag(**{name if kind == "equal_time" else f"X{m}": 1})
        if p.common_rate:
            routes = [("gauss_legendre", dict(n_gauss=8), tuples),
                      ("qmc_vectorized", dict(n_samples=2 ** 16, seed=1),
                       tuples[:4])]
        else:                           # matrix R: the scalar loops only
            routes = [("qmc_scalar", dict(n_samples=2 ** 14, seed=1),
                       tuples[:4])]
        if kind == "equal_time":
            routes.append(("nquad", {}, tuples[:4]))
        for method, kw, tt in routes:
            t0 = time.perf_counter()
            for comps in tt:
                ref = md.split_sum(comps, ts, p, [m],
                                   equal_time=(kind == "equal_time"),
                                   tensors=tensors)
                got = exp.evaluate(props, positions=POS, t_final=max(ts),
                                   component_pair=comps, orders=[1],
                                   external_times=ext, method=method,
                                   **kw).total
                rows.append(dict(noise=f"static {kind}", m=m, route=name,
                                 method=method, times="distinct",
                                 rates=list(p.rates), comps=list(comps),
                                 package=got, reference=ref,
                                 rel=rel(got, ref)))
            print(f"  static {kind:10s} m={m} rates {p.rates} "
                  f"{method:15s} {len(tt):2d} tuples "
                  f"({time.perf_counter() - t0:.1f} s)")
        for comps in tuples[:4]:        # the hierarchy, same configuration
            ref = md.split_sum(comps, ts, p, [m],
                               equal_time=(kind == "equal_time"),
                               tensors=tensors)
            got = H.moments([(c, i) for i, c in enumerate(comps)], [tg],
                            ts)[tg]
            rows.append(dict(noise=f"static {kind}", m=m, route="hierarchy",
                             method="expm", times="distinct",
                             rates=list(p.rates), comps=list(comps),
                             package=got, reference=ref, rel=rel(got, ref)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="high_cumulants_results.json")
    ap.add_argument("--orders", default="4,5")
    args = ap.parse_args()
    orders = [int(v) for v in args.orders.split(",")]
    rows: list = []
    for m in orders:
        for pp in (nz.PARAMS_WHITE, nz.PARAMS_EXP):
            poisson_routes(m, pp, rows)
            poisson_hierarchy(m, pp, rows)
        for p in (md.PARAMS_COMMON, md.PARAMS):
            static_routes(m, p, rows)
    worst: dict = {}
    for r in rows:
        key = (f"{r['noise']} | m={r['m']} | {r['route']} | {r['method']} | "
               f"{r['times']} times"
               + (f" | rates {r['rates']}" if "rates" in r else ""))
        worst[key] = max(worst.get(key, 0.0), r["rel"])
    for key, val in sorted(worst.items()):
        print(f"  {key:70s} {val:.1e}")
    with open(args.out, "w") as fh:
        json.dump(dict(positions=POS, t=T, unequal=UNEQUAL,
                       static_times=STATIC_TIMES, worst=worst, rows=rows),
                  fh, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
