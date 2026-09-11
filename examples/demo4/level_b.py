r"""Demo 4, level B: the interacting field against the exact moment hierarchy.

``⟨φ_a(x, T) φ_b(y, T)⟩`` for all four component pairs, channel by
channel.  The hierarchy's coefficient of ``ε^k μ^j`` (``k`` powers of F,
``j = Σ (m_i − 2)`` over the non-local vertices) is one channel of the
package:

========  ========  ==========================================
tag       package   diagrams
========  ========  ==========================================
(0, 0)    order 0   the C propagator
(1, 1)    FK3       one F and one κ³ (order 2)
(2, 0)    F         two F, Gaussian noise only (order 2)
(2, 2)    FK4       two F and one κ⁴ (order 3)
========  ========  ==========================================

plus ``⟨φ_a(x, T)⟩`` at ``(1, 0)``, the F tadpole.  No other diagram
contributes to these tags.

The non-local vertices are the ones whose integrands are smooth, so that
Gauss-Legendre converges exponentially:

* exponential pulses: ``already_R_contracted`` (``K_R``); the raw kernel is
  kinked where two leg times cross;
* white pulses: the raw ``equal_time`` vertex, whose time is bounded by its
  partners through the causal mapping; ``K_R`` depends on the smallest
  partner time and is kinked where two F-vertex times cross.  White noise
  also kinks C on its time diagonal; ``integrate_moment_gauss_legendre``
  splits the domain there.

Run ``python level_b.py``; results go to ``level_b_results.json``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

os.environ.setdefault("SFT_WICK_QUIET_CACHE", "1")

import poisson_noise as nz  # noqa: E402
import poisson_system as dsys  # noqa: E402
from poisson_reference import Hierarchy  # noqa: E402

T = 1.7
POS = {"x": 0.0, "y": 0.8}
CHANNELS = [("order 0", 0, None, (0, 0)), ("FK3", 2, "FK3", (1, 1)),
            ("FF", 2, "F", (2, 0)), ("FFK4", 3, "FK4", (2, 2))]
N_GAUSS = {"white": {2: 16, 3: 16}, "exponential": {2: 32, 3: 24}}


def run(p, quadrature: bool = False):
    H = Hierarchy(p, [POS["x"], POS["y"]])
    system = dsys.make_system(p, f_amplitude=1.0, cumulants=(3, 4),
                              r_contracted=(p.pulse == "exponential"))
    props = (dsys.quadrature_propagators_for(
                 system, p, t_max=T + 0.5,
                 n_grid_t=21 if p.pulse == "white" else 17)
             if quadrature else dsys.propagators_for(system, p, t_max=T + 0.5))
    two = system.expand(("phi_a(x)", "phi_b(y)"), orders=[0, 2, 3],
                        diag_C=False)
    one = system.expand(("phi_a(x)",), orders=[1], diag_C=False)
    rows = []
    for a, b in itertools.product(range(2), repeat=2):
        for name, order, vtype, tag in CHANNELS:
            ref = H.moments([(a, 0), (b, 1)], [tag], T)[tag]
            kw = dict(orders=[order], method="gauss_legendre",
                      n_gauss=N_GAUSS[p.pulse].get(order, 8))
            if vtype:
                kw["vertex_types"] = {vtype}
            got = two.evaluate(props, positions=POS, t_final=T,
                               component_pair=(a, b), **kw).total
            rows.append(dict(obs=f"<phi_{a}(x) phi_{b}(y)>", channel=name,
                             tag=list(tag), package=got, hierarchy=ref))
    for a in range(2):
        ref = H.moments([(a, 0)], [(1, 0)], T)[(1, 0)]
        got = one.evaluate(props, positions=POS, t_final=T,
                           component_pair=(a,), orders=[1],
                           method="gauss_legendre", n_gauss=32).total
        rows.append(dict(obs=f"<phi_{a}(x)>", channel="F tadpole",
                         tag=[1, 0], package=got, hierarchy=ref))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="level_b_results.json")
    ap.add_argument("--c-quadrature", action="store_true",
                    help="tabulate C by quadrature instead of using the "
                         "closed form of poisson_noise; accuracy is then "
                         "the table's")
    args = ap.parse_args()
    out = {}
    for p in (nz.PARAMS_WHITE, nz.PARAMS_EXP):
        t0 = time.perf_counter()
        rows = run(p, quadrature=args.c_quadrature)
        secs = time.perf_counter() - t0
        print(f"\n{p.pulse} pulses ({secs:.1f} s, n_gauss "
              f"{N_GAUSS[p.pulse]})")
        print(f"  {'observable':>18} {'channel':>9} {'package':>16} "
              f"{'hierarchy':>16} {'rel. diff':>9}")
        worst = {}
        for r in rows:
            rel = abs(r["package"] - r["hierarchy"]) / abs(r["hierarchy"])
            r["rel"] = rel
            worst[r["channel"]] = max(worst.get(r["channel"], 0.0), rel)
            print(f"  {r['obs']:>18} {r['channel']:>9} {r['package']:+16.9e} "
                  f"{r['hierarchy']:+16.9e} {rel:9.1e}")
        print("  worst per channel: "
              + ", ".join(f"{k} {v:.1e}" for k, v in worst.items()))
        out[p.pulse] = dict(rows=rows, worst=worst, seconds=round(secs, 1),
                            n_gauss=N_GAUSS[p.pulse])
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
