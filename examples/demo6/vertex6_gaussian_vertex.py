r"""Demo 6, part F: a non-local ``m = 2`` vertex, Gaussian noise entered as
a vertex.

``NonLocalVertex(order=2)`` carries the MSR factor ``−i²/2! = ½``, so its
order-1 contribution to ``⟨φ_a(x, t_x) φ_b(y, t_y)⟩`` is
``∫∫ R κ² R``, the C propagator of that noise, exactly.  Four kernels, one
per route:

=======================  ==================================================
route                    ``κ²`` and its C
=======================  ==================================================
static (ndarray)         the static force's covariance; ``C = κ² g g``
``equal_time`` (ndarray) a white noise, ``C = W ∫ e^{−γ_a(t_x−s)}
                         e^{−γ_b(t_y−s)} ds``
R-contracted callable    demo 4's compound-Poisson noise: ``K_R`` at
                         ``m = 2`` is its C
raw callable             the same, with the package doing the two leg
                         integrals (``equal_time`` for white pulses)
=======================  ==================================================

With ``F ≠ 0`` the vertex enters channels of its own, which the moment
hierarchy tags: ``F K2`` in ``⟨φ_a(x)⟩`` at order 2 and ``F F K2`` in
``⟨φ_a(x) φ_b(y)⟩`` at order 3, for the static and the white vertex.

Run ``python vertex6_gaussian_vertex.py``; results go to
``gaussian_vertex_results.json``.
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
import vertex6_model as md  # noqa: E402
import vertex6_system as vs  # noqa: E402
from vertex6_reference import Hierarchy, tag  # noqa: E402

POS = {"x": 0.0, "y": 0.8}
TIMES = {"x": 1.9, "y": 1.3}
#: demo 4's noise is not stationary before t = 0 for white pulses, and its
#: closed forms start there, so its part runs at its own times.
D4_TIMES = {"x": 1.6, "y": 1.1}
PAIRS = list(itertools.product(range(2), repeat=2))


def rel(got, ref):
    return abs(got - ref) / abs(ref)


def order_one_static(rows):
    """The static and the white ``m = 2`` vertex against their C."""
    for p in (md.PARAMS_COMMON, md.PARAMS):
        system = vs.make_system(p, static=(2,), equal_time=(2,))
        props = vs.propagators_for(system, p, t_max=max(TIMES.values()) + 0.5)
        exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[1],
                            diag_C=False)
        routes = ([("gauss_legendre", dict(n_gauss=8)), ("nquad", {}),
                   ("qmc_vectorized", dict(n_samples=2 ** 14, seed=1))]
                  if p.common_rate
                  else [("qmc_scalar", dict(n_samples=2 ** 14, seed=1)),
                        ("qmc", dict(n_samples=2 ** 14, seed=1)),
                        ("nquad", {})])
        for method, kw in routes:
            for label in ("X2", "W2"):
                for ab in PAIRS:
                    if label == "X2":
                        ref = (md.X_CUMULANTS[2][ab]
                               * md.g_factor(ab[0], TIMES["x"], p)
                               * md.g_factor(ab[1], TIMES["y"], p))
                    else:
                        ref = (md.JUMP_CUMULANTS[2][ab]
                               * md.equal_time_block(
                                   ab, [TIMES["x"], TIMES["y"]], p))
                    got = exp.evaluate(
                        props, positions=POS, t_final=max(TIMES.values()),
                        component_pair=ab, orders=[1], vertex_types={label},
                        external_times=TIMES, method=method, **kw).total
                    rows.append(dict(part="order 1", kernel=label,
                                     rates=list(p.rates), route=method,
                                     comps=list(ab), package=got,
                                     reference=float(ref),
                                     rel=rel(got, float(ref))))


def order_one_poisson(rows):
    """demo 4's compound-Poisson ``κ²`` as a vertex, against its C."""
    for pp in (nz.PARAMS_WHITE, nz.PARAMS_EXP):
        for rc in (True, False):
            system = dsys.make_system(pp, cumulants=(2,), r_contracted=rc)
            props = dsys.propagators_for(
                system, pp, t_max=max(D4_TIMES.values()) + 0.5)
            exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[1],
                                diag_C=False)
            if rc or pp.pulse == "white":
                routes = [("gauss_legendre", dict(n_gauss=16))]
            else:                      # kinked where the two leg times cross
                routes = [("gauss_legendre", dict(n_gauss=32)),
                          ("qmc_vectorized", dict(n_samples=2 ** 16, seed=1))]
            for method, kw in routes:
                for ab in PAIRS:
                    ref = nz.C_matrix(POS["x"], D4_TIMES["x"], POS["y"],
                                      D4_TIMES["y"], pp)[0][ab]
                    got = exp.evaluate(
                        props, positions=POS,
                        t_final=max(D4_TIMES.values()), component_pair=ab,
                        orders=[1], vertex_types={"K2"},
                        external_times=D4_TIMES, method=method, **kw).total
                    rows.append(dict(
                        part="order 1", kernel=f"{pp.pulse} "
                        f"{'R-contracted' if rc else 'raw'}",
                        rates=[pp.gamma] * 2, route=method, comps=list(ab),
                        package=got, reference=float(ref),
                        rel=rel(got, float(ref))))


def with_F(rows):
    """``F K2`` and ``F F K2`` against the hierarchy's tags."""
    p = md.PARAMS_COMMON
    system = vs.make_system(p, f_amplitude=1.0, static=(2,), equal_time=(2,))
    props = vs.propagators_for(system, p, t_max=max(TIMES.values()) + 0.5)
    one = system.expand(("phi_a(x)",), orders=[2], diag_C=False)
    two = system.expand(("phi_a(x)", "phi_b(y)"), orders=[3], diag_C=False)
    sig = lambda a, x, b, y: md.sigma2(a, x, b, y, p)          # noqa: E731
    H1 = Hierarchy(p, [POS["x"]], sigma2=sig, f_tensor=md.F_TENSOR,
                   x_cumulants={2: md.X_CUMULANTS[2]},
                   white_vertex=md.JUMP_CUMULANTS[2])
    H2 = Hierarchy(p, [POS["x"], POS["y"]], sigma2=sig, f_tensor=md.F_TENSOR,
                   x_cumulants={2: md.X_CUMULANTS[2]},
                   white_vertex=md.JUMP_CUMULANTS[2])
    equal = {"x": TIMES["x"], "y": TIMES["x"]}
    cases = [
        ("F K2", one, H1, 2, {"F": 1, "X2": 1}, [(0,), (1,)], {"x": TIMES["x"]}),
        ("F W2", one, H1, 2, {"F": 1, "W2": 1}, [(0,), (1,)], {"x": TIMES["x"]}),
        ("F F K2 (equal times)", two, H2, 3, {"F": 2, "X2": 1},
         [(0, 1), (1, 1)], equal),
        ("F F W2 (equal times)", two, H2, 3, {"F": 2, "W2": 1},
         [(0, 1), (1, 1)], equal),
        ("F F K2 (distinct times)", two, H2, 3, {"F": 2, "X2": 1},
         [(0, 1), (1, 1)], dict(TIMES)),
        ("F F W2 (distinct times)", two, H2, 3, {"F": 2, "W2": 1},
         [(0, 1), (1, 1)], dict(TIMES)),
    ]
    for name, exp, H, order, counts, tuples, ext in cases:
        sub = vs.restrict(exp, order, counts)
        tg = tag(**counts)
        routes = ([("gauss_legendre", dict(n_gauss=12))] if "distinct"
                  not in name else
                  [("gauss_legendre", dict(n_gauss=16)),
                   ("qmc_vectorized", dict(n_samples=2 ** 18, seed=3))])
        for method, kw in routes:
            for comps in tuples:
                times = [ext[k] for k in list("xy")[:len(comps)]]
                ref = H.moments([(c, i) for i, c in enumerate(comps)], [tg],
                                times)[tg]
                got = sub.evaluate(props, positions=POS, t_final=max(times),
                                   component_pair=comps, orders=[order],
                                   external_times=ext, method=method,
                                   **kw).total
                rows.append(dict(part="with F", kernel=name,
                                 rates=list(p.rates), route=method,
                                 comps=list(comps), package=got,
                                 reference=ref, rel=rel(got, ref)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="gaussian_vertex_results.json")
    args = ap.parse_args()
    rows: list = []
    for step in (order_one_static, order_one_poisson, with_F):
        t0 = time.perf_counter()
        step(rows)
        print(f"  {step.__name__}: {time.perf_counter() - t0:.1f} s")
    worst: dict = {}
    for r in rows:
        key = f"{r['part']} | {r['kernel']} | {r['route']} | {r['rates']}"
        worst[key] = max(worst.get(key, 0.0), r["rel"])
    for key, val in worst.items():
        print(f"  {key:60s} {val:.1e}")
    with open(args.out, "w") as fh:
        json.dump(dict(positions=POS, times=TIMES, demo4_times=D4_TIMES,
                       worst=worst, rows=rows), fh, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
