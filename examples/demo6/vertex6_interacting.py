r"""Demo 6, part B: static cumulants with ``F ≠ 0``.

``dφ_a = (−γ_a φ_a + F_abc φ_b φ_c + X_a) dt + dW_a`` with ``X`` a random
vector constant in space and time, entered as static (ndarray) non-local
vertices ``X3`` and ``X4``.  Channels of ``⟨φ_a(x, t_x) φ_b(y, t_y)⟩``:

============  =======  =====================  ===========================
channel       order    composition            what it contains
============  =======  =====================  ===========================
order 0       0        —                      the C propagator
F X3          2        F X3                   no C: an R tree
F F           2        F F                    one C, two F vertices
F F X4        3        F F X4                 no C: an R tree
============  =======  =====================  ===========================

and, with ``--two-copies``, the order-3 ``F X3 X3`` channel of the
five-point function ``⟨φ_a(x) … φ_e(v)⟩`` — two copies of one static vertex
together with an interaction — and its ``equal_time`` counterpart
``F J3 J3``.

Reference: the moment hierarchy (:mod:`vertex6_reference`) at the tag with
the same vertex counts, at the same distinct times.

Two limits of Gauss-Legendre show up here and are reported rather than
worked around.  A kink between an internal time and a **fixed external**
time — the ends of a white-noise C, or two parents of one equal-time vertex
— is not one of the pairs the Gauss-Legendre kink split orders, so at
distinct external times those channels converge algebraically; at equal
external times, and in channels with no C and no equal-time vertex, the
convergence is exponential.  QMC is unaffected.  The order-4 expansion of a
*four*-point observable (which would give ``F F X3 X3``) did not finish in
20 minutes, so the two-copy channel is taken at order 3 of the five-point
function instead.

Run ``python vertex6_interacting.py``; results go to
``interacting_results.json``.
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

POS = {"x": 0.0, "y": 0.8, "z": -0.5, "w": 1.2, "v": 0.3}
TIMES = {"x": 1.9, "y": 1.3, "z": 2.2, "w": 1.6, "v": 2.0}
EQUAL = {"x": 1.9, "y": 1.9}
PAIRS = [(0, 1), (1, 0), (1, 1)]
FIVE = [(0, 1, 1, 0, 1), (1, 1, 0, 1, 0)]

#: (channel, order, composition, external times, routes)
TWO_POINT = [
    ("order 0", 0, {}, "distinct",
     [("common", "gauss_legendre", dict(n_gauss=8))]),
    ("F X3", 2, {"F": 1, "X3": 1}, "distinct",
     [("common", "gauss_legendre", dict(n_gauss=8)),
      ("common", "qmc_vectorized", dict(n_samples=2 ** 16, seed=1)),
      ("distinct", "qmc_scalar", dict(n_samples=2 ** 14, seed=1))]),
    ("F F X4", 3, {"F": 2, "X4": 1}, "distinct",
     [("common", "gauss_legendre", dict(n_gauss=8)),
      ("distinct", "qmc_scalar", dict(n_samples=2 ** 14, seed=1))]),
    ("F F", 2, {"F": 2}, "equal",
     [("common", "gauss_legendre", dict(n_gauss=12))]),
    ("F F", 2, {"F": 2}, "distinct",
     [("common", "gauss_legendre", dict(n_gauss=16)),
      ("common", "qmc_vectorized", dict(n_samples=2 ** 18, seed=3))]),
]
PARAMS = {"common": md.PARAMS_COMMON, "distinct": md.PARAMS}


def rel(got, ref):
    return abs(got - ref) / abs(ref)


def two_point(rows):
    setups = {}
    hier = {}
    for variant, p in PARAMS.items():
        system = vs.make_system(p, f_amplitude=1.0, static=(3, 4))
        props = vs.propagators_for(system, p, t_max=max(TIMES.values()) + 0.5)
        exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[0, 2, 3],
                            diag_C=False)
        setups[variant] = (p, props, exp)
        hier[variant] = Hierarchy(
            p, [POS["x"], POS["y"]], f_tensor=md.F_TENSOR,
            sigma2=lambda a, x, b, y, p=p: md.sigma2(a, x, b, y, p),
            x_cumulants={3: md.X_CUMULANTS[3], 4: md.X_CUMULANTS[4]})
    for name, order, counts, times, routes in TWO_POINT:
        ext = EQUAL if times == "equal" else {k: TIMES[k] for k in "xy"}
        for variant, method, kw in routes:
            p, props, exp = setups[variant]
            sub = exp if order == 0 else vs.restrict(exp, order, counts)
            tg = tag(**counts)
            t0 = time.perf_counter()
            for ab in PAIRS:
                ref = hier[variant].moments([(ab[0], 0), (ab[1], 1)], [tg],
                                            [ext["x"], ext["y"]])[tg]
                got = sub.evaluate(props, positions=POS,
                                   t_final=max(ext.values()),
                                   component_pair=ab, orders=[order],
                                   external_times=ext, method=method,
                                   **kw).total
                rows.append(dict(observable="2pt", channel=name, order=order,
                                 times=times, rates=list(p.rates),
                                 route=method, comps=list(ab), package=got,
                                 hierarchy=ref, rel=rel(got, ref)))
            print(f"  2pt {name:7s} {times:8s} rates {p.rates} "
                  f"{method:15s} ({time.perf_counter() - t0:.1f} s)")


def two_copies(rows):
    """``F X3 X3`` and ``F J3 J3`` of the five-point function at order 3."""
    p = md.PARAMS_COMMON
    labels = ("x", "y", "z", "w", "v")
    obs = tuple(f"phi_{c}({lab})" for c, lab in zip("abcde", labels))
    for kind, name in (("equal_time", "J3"), ("static", "X3")):
        t0 = time.perf_counter()
        system = vs.make_system(p, f_amplitude=1.0,
                                static=(3,) if kind == "static" else (),
                                equal_time=(3,) if kind == "equal_time"
                                else ())
        props = vs.propagators_for(system, p, t_max=max(TIMES.values()) + 0.5)
        exp = system.expand(obs, orders=[3], diag_C=False)
        sub = vs.restrict(exp, 3, {"F": 1, name: 2})
        print(f"  5pt {kind}: {len(sub.diagrams(3))} diagrams of F {name} "
              f"{name} (expansion {time.perf_counter() - t0:.0f} s)")
        H = Hierarchy(
            p, [POS[k] for k in labels], f_tensor=md.F_TENSOR,
            sigma2=lambda a, x, b, y: md.sigma2(a, x, b, y, p),
            **({"x_cumulants": {3: md.X_CUMULANTS[3]}} if kind == "static"
               else {"jump_cumulants": {3: md.JUMP_CUMULANTS[3]}}))
        tg = tag(**{"X3" if kind == "static" else "J3": 2, "F": 1})
        routes = [("qmc_vectorized", dict(n_samples=2 ** 18, seed=1)),
                  ("gauss_legendre", dict(n_gauss=8))]
        for method, kw in routes:
            t0 = time.perf_counter()
            for comps in FIVE:
                ref = H.moments([(c, i) for i, c in enumerate(comps)], [tg],
                                [TIMES[k] for k in labels])[tg]
                got = sub.evaluate(props, positions=POS,
                                   t_final=max(TIMES.values()),
                                   component_pair=comps, orders=[3],
                                   external_times=TIMES, method=method,
                                   **kw).total
                rows.append(dict(observable="5pt", channel=f"F {name} {name}",
                                 order=3, times="distinct",
                                 rates=list(p.rates), route=method,
                                 comps=list(comps), package=got,
                                 hierarchy=ref, rel=rel(got, ref)))
            print(f"  5pt {kind:10s} {method:15s} "
                  f"({time.perf_counter() - t0:.1f} s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="interacting_results.json")
    ap.add_argument("--two-copies", action="store_true",
                    help="also the order-3 F X3 X3 channel of the five-point "
                         "function (its expansion takes about a minute)")
    args = ap.parse_args()
    rows: list = []
    two_point(rows)
    if args.two_copies:
        two_copies(rows)
    worst: dict = {}
    for r in rows:
        key = (f"{r['observable']} {r['channel']} | {r['times']} times | "
               f"{r['route']} | rates {r['rates']}")
        worst[key] = max(worst.get(key, 0.0), r["rel"])
    for key, val in worst.items():
        print(f"  {key:70s} {val:.1e}")
    with open(args.out, "w") as fh:
        json.dump(dict(positions=POS, times=TIMES, equal_times=EQUAL,
                       t_min=md.PARAMS.t_min, worst=worst, rows=rows), fh,
                  indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
