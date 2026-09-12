r"""Demo 6, part C: a cubic and a quartic local vertex in one system.

``dφ_a = (−γ_a φ_a + F_abc φ_b φ_c + G_abcd φ_b φ_c φ_d) dt + dW_a``, N = 2,
no index symmetry in either tensor.  Channels of ``⟨φ_a(x)⟩`` and
``⟨φ_a(x) φ_b(y)⟩`` at orders 2 and 3, each selected by its vertex
composition:

========================  =======  =========================
observable                order    composition
========================  =======  =========================
``⟨φ_a⟩``                 2        F G
``⟨φ_a⟩``                 3        F F F, F G G
``⟨φ_a φ_b⟩``             2        F F, G G
``⟨φ_a φ_b⟩``             3        F F G, G G G
========================  =======  =========================

(The other compositions vanish: a diagram needs as many φ legs as ψ legs
plus an even number for the C propagators.)

Reference: the moment hierarchy of the Markov embedding with a cubic drift
term (:mod:`vertex6_reference`), at the tag that counts the same vertices.

Two external times: the two-point channels are run at equal times and at
distinct times, where a C propagator between an internal time and a *fixed
external* time is kinked inside the domain.  Gauss-Legendre cuts the
internal variable's range at that time (since 2026-09-12) and converges
exponentially in both; at distinct times it used to fall to 8.5e-04.

Run ``python vertex6_cubic.py``; results go to ``cubic_results.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("SFT_WICK_QUIET_CACHE", "1")

import vertex6_model as md  # noqa: E402
import vertex6_system as vs  # noqa: E402
from vertex6_reference import Hierarchy, tag  # noqa: E402

POS = {"x": 0.0, "y": 0.8}
T_EQUAL = 1.9
T_DISTINCT = {"x": 1.9, "y": 1.3}

#: (observable, order, composition, component tuples)
CHANNELS = [
    ("1pt", 2, {"F": 1, "G": 1}, [(0,), (1,)]),
    ("1pt", 3, {"F": 3}, [(0,), (1,)]),
    ("1pt", 3, {"F": 1, "G": 2}, [(0,), (1,)]),
    ("2pt", 2, {"F": 2}, [(0, 1), (1, 0), (1, 1)]),
    ("2pt", 2, {"G": 2}, [(0, 1), (1, 0), (1, 1)]),
    ("2pt", 3, {"F": 2, "G": 1}, [(0, 1), (1, 1)]),
    ("2pt", 3, {"G": 3}, [(0, 1), (1, 1)]),
]
#: (name, times, method, kwargs) — "equal" and "distinct" external times.
#: The order-3 two-point channels have 68 (``F F G``) and 44 (``G G G``)
#: diagrams, so they take the cheaper settings of :data:`ROUTES_ORDER3`.
ROUTES = [
    ("GL equal times", "equal", "gauss_legendre", dict(n_gauss=12)),
    ("GL distinct times", "distinct", "gauss_legendre", dict(n_gauss=16)),
    ("QMC distinct times", "distinct", "qmc_vectorized",
     dict(n_samples=2 ** 18, seed=3)),
]
ROUTES_ORDER3 = [
    ("GL equal times", "equal", "gauss_legendre", dict(n_gauss=10)),
    ("GL distinct times", "distinct", "gauss_legendre", dict(n_gauss=10)),
    ("QMC distinct times", "distinct", "qmc_vectorized",
     dict(n_samples=2 ** 14, seed=3)),
]


def rel(got, ref):
    return abs(got - ref) / abs(ref)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="cubic_results.json")
    ap.add_argument("--fast", action="store_true",
                    help="skip the order-3 two-point channels")
    args = ap.parse_args()

    p = md.PARAMS_COMMON
    system = vs.make_system(p, f_amplitude=1.0, g_amplitude=1.0)
    props = vs.propagators_for(system, p, t_max=T_EQUAL + 0.5)
    exps = {"1pt": system.expand(("phi_a(x)",), orders=[2, 3], diag_C=False),
            "2pt": system.expand(("phi_a(x)", "phi_b(y)"), orders=[2, 3],
                                 diag_C=False)}
    hier = {
        "1pt": Hierarchy(p, [POS["x"]], f_tensor=md.F_TENSOR,
                         g_tensor=md.G_TENSOR,
                         sigma2=lambda a, x, b, y: md.sigma2(a, x, b, y, p)),
        "2pt": Hierarchy(p, [POS["x"], POS["y"]], f_tensor=md.F_TENSOR,
                         g_tensor=md.G_TENSOR,
                         sigma2=lambda a, x, b, y: md.sigma2(a, x, b, y, p)),
    }
    rows = []
    worst: dict = {}
    for obs, order, counts, tuples in CHANNELS:
        if args.fast and obs == "2pt" and order == 3:
            continue
        sub = vs.restrict(exps[obs], order, counts)
        tg = tag(**counts)
        name = " ".join(f"{k}{'' if v == 1 else v}" for k, v in counts.items())
        routes = (ROUTES_ORDER3 if (obs == "2pt" and order == 3) else ROUTES)
        for route, times, method, kw in routes:
            if obs == "1pt" and times == "distinct":
                continue          # one external point: no second time
            ext = ({"x": T_EQUAL, "y": T_EQUAL} if times == "equal"
                   else dict(T_DISTINCT))
            if obs == "1pt":
                ext = {"x": T_EQUAL}
            t0 = time.perf_counter()
            for comps in tuples:
                ref = hier[obs].moments(
                    [(c, i) for i, c in enumerate(comps)], [tg],
                    [ext[k] for k in list("xy")[:len(comps)]])[tg]
                got = sub.evaluate(props, positions=POS,
                                   t_final=max(ext.values()),
                                   component_pair=comps, orders=[order],
                                   external_times=ext, method=method,
                                   **kw).total
                rows.append(dict(observable=obs, order=order, channel=name,
                                 route=route, comps=list(comps),
                                 package=got, hierarchy=ref,
                                 rel=rel(got, ref)))
            key = (name, route)
            secs = time.perf_counter() - t0
            worst[f"{name} | {route}"] = max(
                r["rel"] for r in rows if (r["channel"], r["route"]) == key)
            print(f"  {obs} order {order} {name:10s} {route:20s} "
                  f"{len(tuples)} tuples  worst {worst[f'{name} | {route}']:.1e}"
                  f"  ({secs:.1f} s)")
    with open(args.out, "w") as fh:
        json.dump(dict(positions=POS, t_equal=T_EQUAL,
                       t_distinct=T_DISTINCT, t_min=p.t_min,
                       rates=list(p.rates), worst=worst, rows=rows), fh,
                  indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
