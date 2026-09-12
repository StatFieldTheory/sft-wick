r"""Demo 4, level B at order 4: the ``F³κ³`` channel of ``⟨φ_a(x) φ_b(y)⟩``.

Level B (``level_b.py``) stops at order 3.  The next channel is order 4 with
three ``F`` vertices and one ``κ³``: 30 diagrams, one C propagator, and the
moment hierarchy's coefficient of ``ε³ μ¹`` (tag ``(3, 1)``) is exactly
it.  Demo 2 could only estimate the same channel of its own system.

The other order-4 composition of this observable is ``F⁴`` (64 diagrams,
tag ``(4, 0)``); ``F²κ³κ³`` and the rest need more φ legs than the
observable has, so ``vertex_types={"FK3"}`` at order 4 is the ``F³κ³``
channel alone.  The script checks that.

Integrators, as in ``level_b.py``: white pulses take the raw
``equal_time`` κ³, whose integrand is smooth, and Gauss-Legendre converges
exponentially; exponential pulses take the R-contracted ``K_R``, which is
kinked where two partner times cross, and converge algebraically.

Run ``python poisson_level_b_order4.py``; results go to
``level_b_order4_results.json``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from collections import Counter

os.environ.setdefault("SFT_WICK_QUIET_CACHE", "1")

import poisson_noise as nz  # noqa: E402
import poisson_system as dsys  # noqa: E402
from poisson_reference import Hierarchy  # noqa: E402

T = 1.7
POS = {"x": 0.0, "y": 0.8}
TAG = (3, 1)
#: nodes per dimension; the convergence table below is measured at these.
N_GAUSS = {"white": 12, "exponential": 24}
CONVERGENCE = {"white": (8, 12, 16), "exponential": (12, 16, 24, 32)}


def composition(dt) -> Counter:
    """``{vertex name: count}`` of one diagram, from one coupling term."""
    from sft_wick.expressions import Product, Sum, Symbol

    def names(expr):
        if isinstance(expr, Symbol):
            return [expr.name]
        if isinstance(expr, Product):
            return [n for f in expr.factors for n in names(f)]
        return []

    expr = dt.coupling_sum
    return Counter(names(expr.terms[0] if isinstance(expr, Sum) else expr))


def setup(p):
    system = dsys.make_system(p, f_amplitude=1.0, cumulants=(3,),
                              r_contracted=(p.pulse == "exponential"))
    props = dsys.propagators_for(system, p, t_max=T + 0.5)
    exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[4], diag_C=False)
    channel = exp.by_vertex_type(4)["FK3"]
    comps = {tuple(sorted(composition(dt).items())) for dt in channel}
    if comps != {(("F", 3), ("K3", 1))}:
        raise AssertionError(f"the FK3 channel at order 4 is not F^3 K3: "
                             f"{sorted(comps)}")
    return props, exp, len(channel)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="level_b_order4_results.json")
    args = ap.parse_args()
    out = {}
    for p in (nz.PARAMS_WHITE, nz.PARAMS_EXP):
        t0 = time.perf_counter()
        props, exp, n_diagrams = setup(p)
        H = Hierarchy(p, [POS["x"], POS["y"]])
        rows = []
        for a, b in itertools.product(range(2), repeat=2):
            ref = H.moments([(a, 0), (b, 1)], [TAG], T)[TAG]
            got = exp.evaluate(props, positions=POS, t_final=T,
                               component_pair=(a, b), orders=[4],
                               vertex_types={"FK3"}, method="gauss_legendre",
                               n_gauss=N_GAUSS[p.pulse]).total
            rows.append(dict(obs=f"<phi_{a}(x) phi_{b}(y)>", package=got,
                             hierarchy=ref, rel=abs(got - ref) / abs(ref)))
        table = []
        ref01 = H.moments([(0, 1)], [TAG], T)[TAG] if False else rows[1][
            "hierarchy"]
        for n in CONVERGENCE[p.pulse]:
            val = exp.evaluate(props, positions=POS, t_final=T,
                               component_pair=(0, 1), orders=[4],
                               vertex_types={"FK3"}, method="gauss_legendre",
                               n_gauss=n).total
            table.append(dict(n_gauss=n, value=val,
                              rel=abs(val - ref01) / abs(ref01)))
        secs = time.perf_counter() - t0
        out[p.pulse] = dict(n_gauss=N_GAUSS[p.pulse], n_diagrams=n_diagrams,
                            rows=rows, convergence=table,
                            worst=max(r["rel"] for r in rows),
                            seconds=round(secs, 1))
        print(f"\n{p.pulse} pulses, {n_diagrams} diagrams, n_gauss "
              f"{N_GAUSS[p.pulse]} ({secs:.1f} s)")
        for r in rows:
            print(f"  {r['obs']:>20} {r['package']:+16.9e} "
                  f"{r['hierarchy']:+16.9e} {r['rel']:9.1e}")
        print("  convergence (a,b)=(0,1): "
              + ", ".join(f"{t['n_gauss']}: {t['rel']:.1e}" for t in table))
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
