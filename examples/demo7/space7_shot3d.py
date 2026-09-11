r"""Demo 7, item (d): 3-D positions with a callable κ³.

Compound-Poisson shot noise in ``R^3`` (:mod:`space7_shot`): the events'
Gaussian envelopes make every cumulant a Gaussian overlap integral in closed
form, so the ``κ³`` vertex is a callable that receives 3-D leg positions
(``(m, n_samples, 3)`` under the vectorised contract) and the C propagator is
an exact closed form of the 3-vector separation.

Three pulse variants:

===============  =========================  ====================================
variant          ``κ³`` vertex              R
===============  =========================  ====================================
white            raw, ``equal_time``        scalar (one γ)
exponential      ``already_R_contracted``   scalar (one γ)
white-matrixR    raw, ``equal_time``        matrix (γ per component)
===============  =========================  ====================================

Observables: ``⟨φ_a(x) φ_b(y)⟩`` at order 0 (the C propagator) and order 2
(channels ``F`` and ``FK3``), and ``⟨φ_a(x) φ_b(y) φ_c(z)⟩`` at order 1
(channels ``F`` and ``K3``), at off-diagonal component tuples.  The reference
is the moment hierarchy of the Markov process at the three points
(:mod:`space7_shot`, no sft-wick code).

Run ``python space7_shot3d.py``; results go to ``shot3d_results.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("SFT_WICK_QUIET_CACHE", "1")

import numpy as np  # noqa: E402

import space7_shot as sh  # noqa: E402
import space7_shot_system as ss  # noqa: E402
from space7_run import GL, QS, QV, compare  # noqa: E402

#: Observation time after ``t_min``.
T = 1.4
POINTS = {k: np.array(v) for k, v in sh.POINTS.items()}
TWO = ("phi_a(x)", "phi_b(y)")
THREE = ("phi_a(x)", "phi_b(y)", "phi_c(z)")

PULSES = {"white": sh.PARAMS_WHITE,
          "exponential": sh.PARAMS_EXP,
          "white-matrixR": sh.PARAMS_WHITE_MATRIX}
ROUTES = {"white": [GL(16), QV(14), QS(11)],
          "exponential": [GL(16), QV(14), QS(11)],
          "white-matrixR": [QS(11), GL(16), QV(14)]}


def channel_tag(label: str, order: int) -> tuple:
    """``(k_F, j)`` of a package channel: ``j`` counts ``Σ (m_i − 2)`` over
    the non-local vertices, so the FK3 channel is ``(1, 1)``."""
    return {"": (0, 0), "F": (order, 0), "FK3": (1, 1), "K3": (0, 1)}[label]


def run(variant: str, pairs=((0, 1), (1, 0), (1, 1)),
        triples=((0, 1, 1), (1, 0, 1), (1, 1, 0)), routes=None) -> list:
    p = PULSES[variant]
    routes = ROUTES[variant] if routes is None else routes
    system = ss.make_system(p, f_amplitude=1.0, cumulants=(3,))
    props = ss.propagators_for(system, p, t_max=p.t_min + T + 0.3)
    two = system.expand(TWO, orders=[0, 2], diag_C=False, progress=False)
    three = system.expand(THREE, orders=[1], diag_C=False, progress=False)
    H2 = sh.ShotHierarchy(p, [POINTS["x"], POINTS["y"]])
    H3 = sh.ShotHierarchy(p, [POINTS["x"], POINTS["y"], POINTS["z"]])
    jobs = []
    for ab in pairs:
        for order in (0, 2):
            jobs.append((two, TWO, ab, order, H2))
    for abc in triples:
        jobs.append((three, THREE, abc, 1, H3))
    rows = []
    for expansion, observable, comps, order, H in jobs:
        tags = {0: [(0, 0)], 1: [(1, 0), (0, 1)], 2: [(2, 0), (1, 1)]}[order]
        ref = H.moments([(c, i) for i, c in enumerate(comps)], tags, T)
        for method, kw in routes:
            row = dict(variant=variant, observable=" ".join(observable),
                       comps=list(comps), order=order, method=method,
                       settings=dict(kw))
            t0 = time.perf_counter()
            try:
                res = expansion.evaluate(
                    props, positions=POINTS, t_final=p.t_min + T,
                    component_pair=comps, orders=[order], method=method, **kw)
            except NotImplementedError as exc:
                row.update(status="refused", reason=str(exc).split(".")[0])
                rows.append(row)
                continue
            row.update(status="ok", seconds=round(time.perf_counter() - t0, 2),
                       **compare(res, ref, order, channel_tag=channel_tag))
            rows.append(row)
    return rows


def summarise(rows) -> dict:
    out: dict = {}
    for r in rows:
        key = f"{r['variant']} | {r['method']} | order {r['order']}"
        if r["observable"].count("phi") == 3:
            key += " | 3-point"
        e = out.setdefault(key, {"worst": 0.0, "n": 0, "refused": 0})
        if r["status"] == "ok":
            e["worst"] = max(e["worst"], r["error"])
            e["n"] += 1
        else:
            e["refused"] += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", default=",".join(PULSES))
    ap.add_argument("--out", default="shot3d_results.json")
    args = ap.parse_args()
    rows, timing = [], {}
    for variant in args.variants.split(","):
        t0 = time.perf_counter()
        rows += run(variant)
        timing[variant] = round(time.perf_counter() - t0, 1)
        print(f"{variant}: {timing[variant]} s")
    summary = summarise(rows)
    for key, e in summary.items():
        worst = f"{e['worst']:.1e}" if e["n"] else "-"
        print(f"  {key:48s} worst {worst:>8s}  n {e['n']:3d}  "
              f"refused {e['refused']}")
    with open(args.out, "w") as fh:
        json.dump(dict(summary=summary, seconds=timing, rows=rows), fh,
                  indent=1, default=float)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
