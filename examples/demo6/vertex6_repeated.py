r"""Demo 6, part A: two copies of one non-local vertex.

With ``F = G = 0`` the order-2 contribution to the six-point function
``⟨φ_{a_1}(x_1, t_1) … φ_{a_6}(x_6, t_6)⟩`` with two copies of a cubic
vertex is the part of the moment that is a product of two third cumulants:

    ``Σ_{A ∪ B} κ_{a_A} κ_{a_B} w_A w_B``

over the ten splits of the six external points into two triples.  For the
static force (``X3``, an ndarray coupling) ``w_A = Π_{j∈A} g_{a_j}(t_j)``;
for the white jumps (``J3``, ``equal_time=True``) ``w_A`` is the integral
over the one time the three legs share.  Distinct points and times, and
every component tuple.

References, independent of each other and of the package: the closed form
(:func:`vertex6_model.split_sum`) and the moment hierarchy of the Markov
embedding at six observation times (:mod:`vertex6_reference`), at tags
``X3 = 2`` and ``J3 = 2``.

Routes: Gauss-Legendre, ``qmc_vectorized`` and ``nquad`` with one rate for
both components (scalar R); ``qmc_scalar`` and ``qmc`` with distinct rates
(diagonal matrix R, which Gauss-Legendre and ``qmc_vectorized`` refuse
today; ``--pending`` tries them).

Run ``python vertex6_repeated.py``; results go to ``repeated_results.json``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np

os.environ.setdefault("SFT_WICK_QUIET_CACHE", "1")

import vertex6_model as md  # noqa: E402
import vertex6_system as vs  # noqa: E402
from vertex6_reference import Hierarchy, tag  # noqa: E402

LABELS = ("x1", "x2", "x3", "x4", "x5", "x6")
POS = dict(zip(LABELS, (0.0, 0.7, -0.4, 1.1, -0.9, 0.3)))
#: Package times (``t_min = 0.4``).
TIMES = dict(zip(LABELS, (1.9, 1.4, 2.3, 0.9, 1.6, 2.1)))
OBS = tuple(f"phi_{c}({lab})" for c, lab in zip("abcdef", LABELS))
ALL_TUPLES = list(itertools.product(range(2), repeat=6))
FEW_TUPLES = [(0, 1, 1, 0, 1, 0), (1, 1, 0, 0, 0, 1), (1, 0, 1, 1, 0, 1)]

#: (kind, vertex name, tag)
KINDS = {"static": ("X3", tag(X3=2)), "equal_time": ("J3", tag(J3=2))}

#: (params, method, kwargs, tuples) per route and kind.
ROUTES = {
    "static": [
        ("common", "gauss_legendre", dict(n_gauss=8), ALL_TUPLES),
        ("common", "qmc_vectorized", dict(n_samples=2 ** 14, seed=1),
         FEW_TUPLES),
        ("distinct", "qmc_scalar", dict(n_samples=2 ** 16, seed=1),
         FEW_TUPLES),
        ("distinct", "qmc", dict(n_samples=2 ** 16, seed=1), FEW_TUPLES),
    ],
    "equal_time": [
        ("common", "gauss_legendre", dict(n_gauss=12), ALL_TUPLES),
        ("common", "qmc_vectorized", dict(n_samples=2 ** 14, seed=1),
         FEW_TUPLES),
        ("common", "nquad", {}, FEW_TUPLES),
        ("distinct", "qmc_scalar", dict(n_samples=2 ** 11, seed=1),
         FEW_TUPLES),
        ("distinct", "qmc", dict(n_samples=2 ** 11, seed=1), FEW_TUPLES),
        ("distinct", "nquad", {}, FEW_TUPLES[:1]),
    ],
}
PENDING = [("distinct", "gauss_legendre", dict(n_gauss=8)),
           ("distinct", "qmc_vectorized", dict(n_samples=2 ** 12, seed=1))]
PARAMS = {"common": md.PARAMS_COMMON, "distinct": md.PARAMS}


def closed_form(p, kind, comps):
    ts = [TIMES[lab] for lab in LABELS]
    tensors = md.X_CUMULANTS if kind == "static" else md.JUMP_CUMULANTS
    return md.split_sum(comps, ts, p, [3, 3], equal_time=(kind == "equal_time"),
                        tensors=tensors)


def hierarchy(p, kind, tuples):
    name, tg = KINDS[kind]
    kw = ({"x_cumulants": {3: md.X_CUMULANTS[3]}} if kind == "static"
          else {"jump_cumulants": {3: md.JUMP_CUMULANTS[3]}})
    H = Hierarchy(p, [POS[lab] for lab in LABELS],
                  sigma2=lambda a, x, b, y: md.sigma2(a, x, b, y, p), **kw)
    times = [TIMES[lab] for lab in LABELS]
    return {comps: H.moments([(c, i) for i, c in enumerate(comps)], [tg],
                             times)[tg] for comps in tuples}


def setup(p, kind):
    system = vs.make_system(p, static=(3,) if kind == "static" else (),
                            equal_time=(3,) if kind == "equal_time" else ())
    props = vs.propagators_for(system, p, t_max=max(TIMES.values()) + 0.5)
    exp = system.expand(OBS, orders=[2], diag_C=False)
    return props, exp


def evaluate(exp, props, comps, method, kw):
    return exp.evaluate(props, positions=POS, t_final=max(TIMES.values()),
                        component_pair=comps, orders=[2],
                        external_times=TIMES, method=method, **kw).total


def rel(got, ref):
    return abs(got - ref) / abs(ref)


def run_kind(kind, pending=False):
    out = {"routes": [], "pending": []}
    refs = {}
    for variant, p in PARAMS.items():
        tuples = sorted({c for v, _m, _k, tt in ROUTES[kind] if v == variant
                         for c in tt})
        hier = hierarchy(p, kind, tuples)
        closed = {c: closed_form(p, kind, c) for c in tuples}
        refs[variant] = (closed, hier)
        out[f"hierarchy_vs_closed_{variant}"] = max(
            rel(hier[c], closed[c]) for c in tuples)
    setups = {}
    for variant, method, kw, tuples in ROUTES[kind]:
        p = PARAMS[variant]
        if variant not in setups:
            setups[variant] = setup(p, kind)
        props, exp = setups[variant]
        closed, hier = refs[variant]
        t0 = time.perf_counter()
        rows = []
        for comps in tuples:
            got = evaluate(exp, props, comps, method, kw)
            rows.append(dict(comps=list(comps), package=got,
                             closed=closed[comps], hierarchy=hier[comps]))
        secs = time.perf_counter() - t0
        res = dict(
            rates=list(p.rates), method=method, kwargs=kw,
            n_tuples=len(tuples), n_diagrams=len(exp.diagrams(2)),
            vs_closed=max(rel(r["package"], r["closed"]) for r in rows),
            vs_hierarchy=max(rel(r["package"], r["hierarchy"]) for r in rows),
            seconds=round(secs, 1), rows=rows[:4])
        out["routes"].append(res)
        print(f"  {kind:10s} {method:15s} rates {p.rates} "
              f"{len(tuples):3d} tuples  vs closed {res['vs_closed']:.1e}  "
              f"vs hierarchy {res['vs_hierarchy']:.1e}  ({secs:.1f} s)")
    if pending:
        props, exp = setups["distinct"]
        for variant, method, kw in PENDING:
            try:
                got = evaluate(exp, props, FEW_TUPLES[0], method, kw)
                status = (f"ran: vs closed "
                          f"{rel(got, refs['distinct'][0][FEW_TUPLES[0]]):.1e}")
            except NotImplementedError as exc:
                status = f"refused: {str(exc)[:80]}"
            out["pending"].append(dict(method=method, status=status))
            print(f"  pending {kind} {method}: {status}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="repeated_results.json")
    ap.add_argument("--pending", action="store_true",
                    help="also try the routes that refuse a matrix R today")
    args = ap.parse_args()
    summary = {"points": POS, "times": TIMES, "t_min": md.PARAMS.t_min}
    for kind in KINDS:
        summary[kind] = run_kind(kind, pending=args.pending)
        print(f"  {kind}: hierarchy vs closed form "
              f"{summary[kind]['hierarchy_vs_closed_common']:.1e} (common), "
              f"{summary[kind]['hierarchy_vs_closed_distinct']:.1e} "
              f"(distinct)")
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
