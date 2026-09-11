r"""Demo 5: white noise, every integrator, against the exact moment hierarchy.

For each noise variant (:data:`white_model.VARIANTS`) the script evaluates

* ``⟨φ_a(x)⟩`` at orders 1 and 3,
* ``⟨φ_a(x) φ_b(y)⟩`` at orders 0, 2 and 4, all four component pairs,
* ``⟨φ_a(x) φ_b(y) φ_c(z)⟩`` at order 1,

at ``t_final = t_min + T`` and compares each order with the coefficient of
``ε^order`` of the moment hierarchy at ``T`` (:mod:`white_reference`).  The
integrators and where they run:

=================  =============================================
gauss_legendre     every observable and order
qmc_vectorized     orders ≤ 2
qmc_scalar, qmc    order 2 of ``⟨φ_0 φ_1⟩``, order 1 of ``⟨φ_a⟩``
nquad              order 1 of ``⟨φ_a⟩`` (adaptive quadrature is slow
                   on the kinked white-noise integrand at order 2)
=================  =============================================

Run ``python run.py``; results go to ``results.json``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

os.environ.setdefault("SFT_WICK_QUIET_CACHE", "1")

import white_model as m5  # noqa: E402
from white_reference import Hierarchy  # noqa: E402

T = 1.6
POS = {"x": 0.0, "y": 0.7, "z": -0.4}
GL = ("gauss_legendre", dict(n_gauss=12))
QMC_V = ("qmc_vectorized", dict(n_samples=2 ** 14, seed=2))
SCALAR = [("qmc_scalar", dict(n_samples=2 ** 11, seed=2)),
          ("qmc", dict(n_samples=2 ** 11, seed=2))]
NQUAD = ("nquad", {})


def _plan():
    """``(observable, order, component tuple, [(method, kwargs)])``."""
    two = ("phi_a(x)", "phi_b(y)")
    one = ("phi_a(x)",)
    three = ("phi_a(x)", "phi_b(y)", "phi_c(z)")
    plan = []
    for a in range(2):
        plan.append((one, 1, (a,), [GL, QMC_V, *SCALAR, NQUAD]))
        plan.append((one, 3, (a,), [GL]))
    for ab in itertools.product(range(2), repeat=2):
        plan.append((two, 0, ab, [GL]))
        methods = [GL, QMC_V] + (SCALAR if ab == (0, 1) else [])
        plan.append((two, 2, ab, methods))
        plan.append((two, 4, ab, [GL]))
    for abc in [(0, 1, 1), (1, 0, 0), (1, 1, 0)]:
        plan.append((three, 1, abc, [GL]))
    return plan


def run(p):
    system = m5.make_system(p)
    t_final = p.t_min + T
    props = m5.propagators_for(system, p, t_max=t_final + 0.5)
    flags = m5.expand_flags(p)
    labels = ("x", "y", "z")
    hier = {n: Hierarchy(p, [POS[l] for l in labels[:n]], m5.F_TENSOR)
            for n in (1, 2, 3)}
    expansions = {}
    rows = []
    for obs, order, comps, methods in _plan():
        key = (obs, order)
        if key not in expansions:
            expansions[key] = system.expand(obs, orders=[order], **flags)
        ref = hier[len(obs)].moments(
            [(c, i) for i, c in enumerate(comps)], [order], T)[order]
        for method, kw in methods:
            t0 = time.perf_counter()
            got = expansions[key].evaluate(
                props, positions=POS, t_final=t_final, component_pair=comps,
                orders=[order], method=method, **kw).total
            rows.append(dict(
                observable="<" + " ".join(f"phi_{c}({l})" for c, l in
                                          zip(comps, labels)) + ">",
                order=order, method=method, package=got, hierarchy=ref,
                rel=abs(got - ref) / abs(ref) if ref else abs(got),
                seconds=round(time.perf_counter() - t0, 2)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()
    out = {}
    for name in m5.VARIANTS:
        p = m5.variant(name)
        t0 = time.perf_counter()
        rows = run(p)
        secs = time.perf_counter() - t0
        worst: dict = {}
        for r in rows:
            k = f"{r['method']}, order {r['order']}"
            worst[k] = max(worst.get(k, 0.0), r["rel"])
        print(f"\n{name}: flags {m5.expand_flags(p)}, t_min {p.t_min}, "
              f"{len(rows)} evaluations, {secs:.0f} s")
        for k in sorted(worst):
            print(f"  worst rel. difference  {k:28s} {worst[k]:.1e}")
        out[name] = dict(rows=rows, worst=worst, seconds=round(secs, 1))
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
