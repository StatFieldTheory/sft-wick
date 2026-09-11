r"""Demo 4, level A: the free field (``F = 0``).

With no interaction the connected ``m``-point function is one diagram, the
``m`` external ``φ``\ s contracted with the ``m`` legs of ``κ^(m)``, and it
equals the R-contracted cumulant exactly:

    ``⟨φ_{a_1}(x_1, t_1) … φ_{a_m}(x_m, t_m)⟩_c = K_R(a; x, t)``.

Three references for the same numbers, independent of one another:

1. the closed form ``ν Π h · X · T̃`` (:mod:`noise`), itself checked against
   direct quadrature in ``tests/test_demo4_asymmetric_noise.py``;
2. the exact moment hierarchy of the process at the observation points
   (:mod:`reference`), at equal times;
3. for the raw route, the package's own leg integrals.

Every component tuple is evaluated, at distinct points, so a coupling
evaluated at the wrong leg order or with the wrong component routing shows
up: the kernel is symmetric only when (component, point) pairs are permuted
together.

Run ``python level_a.py``; results go to ``level_a_results.json``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np

os.environ.setdefault("SFT_WICK_QUIET_CACHE", "1")

import noise as nz  # noqa: E402
import system as dsys  # noqa: E402
from reference import Hierarchy  # noqa: E402

T = 1.7
LABELS = ("x", "y", "z", "w")
POS = {"x": 0.0, "y": 0.8, "z": -0.5, "w": 1.3}
#: Unequal external times for the R-contracted route: each absorbed leg
#: takes its partner's time.
UNEQUAL = {"x": 1.7, "y": 1.2, "z": 0.6}
QUADS = [(0, 1, 1, 0), (1, 1, 1, 1), (0, 0, 1, 1), (1, 0, 0, 0)]


def _expansion(p, m, rc):
    system = dsys.make_system(p, cumulants=(m,), r_contracted=rc)
    props = dsys.propagators_for(system, p, t_max=T + 0.5)
    obs = tuple(f"phi_{c}({lab})" for c, lab in zip("abcd", LABELS[:m]))
    return props, system.expand(obs, orders=[1], diag_C=False)


def _closed(p, comps, labels, times):
    xs = np.array([POS[lab] for lab in labels])
    return float(nz.K_R(comps, xs, np.array(times, float), p)[0])


def _rel(got, ref):
    return abs(got - ref) / abs(ref)


def three_point(p, *, raw_method, raw_kw):
    """All 8 component triples at equal times: package (R-contracted and
    raw) against the closed form and the hierarchy."""
    labels = LABELS[:3]
    H = Hierarchy(p, [POS[lab] for lab in labels])
    rows = []
    pr_rc, ex_rc = _expansion(p, 3, rc=True)
    pr_raw, ex_raw = _expansion(p, 3, rc=False)
    for comps in itertools.product(range(2), repeat=3):
        ref = _closed(p, comps, labels, [T] * 3)
        hier = H.moments([(c, i) for i, c in enumerate(comps)], [(0, 1)],
                         T)[(0, 1)]
        rc = ex_rc.evaluate(pr_rc, positions=POS, t_final=T,
                            component_pair=comps, orders=[1],
                            method="gauss_legendre", n_gauss=8).total
        raw = ex_raw.evaluate(pr_raw, positions=POS, t_final=T,
                              component_pair=comps, orders=[1],
                              method=raw_method, **raw_kw).total
        rows.append(dict(comps=comps, closed=ref, hierarchy=hier,
                         r_contracted=rc, raw=raw))
    return rows


def unequal_times(p):
    """R-contracted route at unequal external times."""
    labels = LABELS[:3]
    pr, ex = _expansion(p, 3, rc=True)
    worst = 0.0
    for comps in itertools.product(range(2), repeat=3):
        ref = _closed(p, comps, labels, [UNEQUAL[lab] for lab in labels])
        got = ex.evaluate(pr, positions=POS, t_final=max(UNEQUAL.values()),
                          component_pair=comps, orders=[1],
                          external_times=UNEQUAL,
                          method="gauss_legendre", n_gauss=8).total
        worst = max(worst, _rel(got, ref))
    return worst


def four_point(p):
    """Connected four-point function (the order-1 κ⁴ diagram) at equal
    times, against the closed form and the hierarchy's ``μ²`` part."""
    labels = LABELS[:4]
    H = Hierarchy(p, [POS[lab] for lab in labels])
    pr, ex = _expansion(p, 4, rc=True)
    rows = []
    for comps in QUADS:
        ref = _closed(p, comps, labels, [T] * 4)
        hier = H.moments([(c, i) for i, c in enumerate(comps)], [(0, 2)],
                         T)[(0, 2)]
        got = ex.evaluate(pr, positions=POS, t_final=T, component_pair=comps,
                          orders=[1], method="gauss_legendre",
                          n_gauss=8).total
        rows.append(dict(comps=comps, closed=ref, hierarchy=hier,
                         r_contracted=got))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="level_a_results.json")
    args = ap.parse_args()
    summary = {}
    for p, raw_method, raw_kw in [
        (nz.PARAMS_WHITE, "gauss_legendre", dict(n_gauss=16)),
        (nz.PARAMS_EXP, "qmc_vectorized", dict(n_samples=2 ** 18, seed=1)),
    ]:
        t0 = time.perf_counter()
        rows3 = three_point(p, raw_method=raw_method, raw_kw=raw_kw)
        rows4 = four_point(p)
        unequal = unequal_times(p)
        res = {
            "3pt_rc_vs_closed": max(_rel(r["r_contracted"], r["closed"]) for r in rows3),
            "3pt_raw_vs_closed": max(_rel(r["raw"], r["closed"]) for r in rows3),
            "3pt_hierarchy_vs_closed": max(_rel(r["hierarchy"], r["closed"]) for r in rows3),
            "3pt_unequal_times_rc_vs_closed": unequal,
            "4pt_rc_vs_closed": max(_rel(r["r_contracted"], r["closed"]) for r in rows4),
            "4pt_hierarchy_vs_closed": max(_rel(r["hierarchy"], r["closed"]) for r in rows4),
            "raw_route": f"{raw_method} {raw_kw}",
            "seconds": round(time.perf_counter() - t0, 1),
            "three_point": [{k: (list(v) if k == "comps" else v)
                             for k, v in r.items()} for r in rows3],
            "four_point": [{k: (list(v) if k == "comps" else v)
                            for k, v in r.items()} for r in rows4],
        }
        summary[p.pulse] = res
        print(f"\n{p.pulse} pulses ({res['seconds']} s)")
        print(f"  {'(a,b,c)':>9} {'closed form':>15} {'R-contracted':>15} "
              f"{'raw':>15} {'hierarchy':>15}")
        for r in rows3:
            print(f"  {str(r['comps']):>9} {r['closed']:+15.8e} "
                  f"{r['r_contracted']:+15.8e} {r['raw']:+15.8e} "
                  f"{r['hierarchy']:+15.8e}")
        for key in ("3pt_rc_vs_closed", "3pt_raw_vs_closed",
                    "3pt_hierarchy_vs_closed",
                    "3pt_unequal_times_rc_vs_closed", "4pt_rc_vs_closed",
                    "4pt_hierarchy_vs_closed"):
            print(f"  max rel {key:32s} {res[key]:.1e}")
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
