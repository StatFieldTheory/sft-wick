r"""Demo 5, part C: multiplicative white noise at L1, Itô and Stratonovich.

.. math::

    dφ_a = (−γ_a φ_a + η_a + F_{abc} φ_b φ_c)\,dt + g_{ak}(φ)\,(\circ)\,dW_k ,
    \qquad g_{ak}(φ) = g^{(0)}_{ak} + g^{(1)}_{akb} φ_b ,\quad k = 0, 1, 2,

one point, ``t ≥ t_min = 0.4`` from ``φ(t_min) = 0``; ``η`` is the
coloured noise of part A (``λ = 0.3``, ``σ_t = 0.7``).  Two components,
three Wiener processes, and no symmetry in ``F``, ``g0`` or ``g1``.

The L1 system declares the white noise as
``MultiplicativeImpulse(g0, g1, interpretation)``.  ``D0 = g0 g0ᵀ`` enters
C through the built-in closed form; the rest of ``D(φ) = g(φ) g(φ)ᵀ``
becomes the local vertices ``G`` (ψψφ) and ``H`` (ψψφφ) with the factor
``½``; the Stratonovich reading adds the noise-induced drift as a source
``B`` (ψ) and a linear vertex ``L`` (ψφ) with the factor ``−i``.

Tags ``(k_F, k_g)`` count powers of ``F`` and ``g1``: ``F`` (1, 0), ``G``
and ``B`` (0, 1), ``H`` and ``L`` (0, 2).  Each vertex has weight
``k_F + k_g ≥ 1``, so a tag of weight ``w`` collects diagrams of vertex
orders up to ``w``.  For every tag of weight ``≤ 3`` the script sums the
package's diagrams of that tag and compares the sum with the tag's
coefficient in the exact hierarchy (``white_hormander_reference.py``),
for ``⟨φ_a⟩`` and ``⟨φ_a φ_b⟩`` (and a two-time ``⟨φ_a(t₁) φ_b(t₂)⟩``).

Equal rates give a scalar R (Gauss-Legendre, ``qmc_vectorized``); distinct
rates a diagonal matrix R, which runs on the scalar loops (``qmc_scalar``,
``nquad``).

With the two external points at **different times**, Gauss-Legendre
converges algebraically: an internal time crosses the earlier external time
inside the domain, where C is kinked (white noise), and the GL kink split
pairs internal variables only.  The two-time part therefore reports
``nquad`` and ``qmc_vectorized`` as well.

Run ``python white_l1_multiplicative.py``; results go to
``white_l1_multiplicative_results.json``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np

os.environ.setdefault("SFT_WICK_QUIET_CACHE", "1")

import sft_wick as sw  # noqa: E402
from sft_wick.expressions import Product, Sum, Symbol  # noqa: E402
from white_hormander_reference import MultiplicativeHierarchy  # noqa: E402
from white_model import F_TENSOR  # noqa: E402

N, M = 2, 3
T_MIN, T = 0.4, 1.5
LAM, SIGMA_T, SIGMA_X = 0.3, 0.7, 0.9
#: g0[a, k]; g1[a, k, b] = ∂g_ak/∂φ_b.  Mixed signs, no symmetry.
G0 = np.array([[0.55, -0.20, 0.10],
               [0.15, 0.40, -0.25]])
G1 = np.array([[[0.20, -0.15], [-0.10, 0.25], [0.05, 0.10]],
               [[-0.12, 0.18], [0.22, -0.08], [-0.15, 0.06]]])
GAMMAS = {"scalar": (1.1, 1.1), "matrix": (0.7, 1.6)}
INTERPRETATIONS = ("ito", "stratonovich")
ONE = ("phi_a(x)",)
TWO = ("phi_a(x)", "phi_b(y)")
THREE = ("phi_a(x)", "phi_b(y)", "phi_c(z)")
#: Every external point at one position: the model is one SDE per point.
POS = {"x": 0.0, "y": 0.0, "z": 0.0}
#: Two-time check: ⟨φ_a(t_min + T1) φ_b(t_min + T)⟩.
T1 = 0.9


def make_system(gammas, interpretation="ito") -> sw.System:
    return sw.System(
        field=sw.FieldSpec("phi", N),
        linear=sw.DiagonalA(gamma=list(gammas)),
        vertices=[sw.LocalVertex("F", coupling=F_TENSOR)],
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=LAM, sigma_t=SIGMA_T),
                spatial=sw.GaussianSpatial(sigma_x=SIGMA_X)),
            sigma2=sw.MultiplicativeImpulse(g0=G0, g1=G1,
                                            interpretation=interpretation)),
        t_min=T_MIN)


def propagators_for(system):
    """C from the built-in closed form: exponential coloured noise plus the
    white noise ``D0 = g0 g0ᵀ``, which mixes the components."""
    return system.propagators(
        t_max=T_MIN + T + 0.5, c_closed_form="auto", c_closed_form_only=True,
        c_closed_form_vectorized=True, diag_C=False, progress=False)


def hierarchy(gammas, interpretation) -> MultiplicativeHierarchy:
    return MultiplicativeHierarchy(gammas, F_TENSOR, G0, G1, LAM, SIGMA_T,
                                   interpretation)


# -- tags of the package's diagrams ------------------------------------------


def weights(system) -> dict[str, tuple[int, int]]:
    """Vertex name → ``(k_F, k_g)``."""
    w = {v.name: (1, 0) for v in system.vertices}
    w.update({v.name: (0, v.g1_power) for v in system.multiplicative_vertices})
    return w


def vertex_counts(expr) -> Counter:
    """The vertex multiset of a diagram: coupling symbols of one term of its
    coupling sum (every term has the same vertices)."""
    counts: Counter = Counter()

    def walk(e):
        if isinstance(e, Symbol):
            counts[e.name] += 1
        elif isinstance(e, Product):
            for f in e.factors:
                walk(f)
        elif isinstance(e, Sum):
            walk(e.terms[0])

    walk(expr)
    return counts


def tag_of(dt, w) -> tuple[int, int]:
    kf = kg = 0
    for name, n in vertex_counts(dt.coupling_sum).items():
        kf += n * w[name][0]
        kg += n * w[name][1]
    return kf, kg


def labels(w, order, max_weight) -> set[str]:
    """Vertex-type labels of the multisets of ``order`` vertices with weight
    ``≤ max_weight`` (what ``Expansion.evaluate(vertex_types=…)`` takes)."""
    out = set()
    for ms in itertools.combinations_with_replacement(sorted(w), order):
        if sum(sum(w[v]) for v in ms) <= max_weight:
            out.add("".join(sorted(set(ms))))
    return out


def expansions(system, obs, max_order) -> dict:
    return {o: system.expand(obs, orders=[o], diag_C=False)
            for o in range(max_order + 1)}


def package_by_tag(system, props, exps, comps, method, kw, max_weight,
                   external_times=None, t_final=T_MIN + T) -> dict:
    """``{tag: Σ of the package's diagrams of that tag}``, weight ``≤
    max_weight``, through ``Expansion.evaluate``."""
    w = weights(system)
    out: dict = defaultdict(float)
    for order in range(max_weight + 1):
        exp = exps[order]
        res = exp.evaluate(
            props, positions=POS, t_final=t_final, component_pair=comps,
            orders=[order], vertex_types=labels(w, order, max_weight),
            method=method, external_times=external_times, **kw)
        for row in res.per_diagram:
            tag = tag_of(exp.diagrams(order)[row["diagram_idx"]], w)
            if sum(tag) <= max_weight:
                out[tag] += row["value"]
    return dict(out)


def all_tags(max_weight):
    return [(kf, kg) for kf in range(max_weight + 1)
            for kg in range(max_weight + 1 - kf)]


def _component_tuples(obs, comps_list):
    if comps_list is not None:
        return [tuple(c) for c in comps_list]
    return list(itertools.product(range(N), repeat=len(obs)))


def compare(interpretation, gammas, method, kw, plan):
    """Rows for each ``(observable, component tuples, max weight)`` of
    ``plan``, at every tag of that weight or below."""
    system = make_system(gammas, interpretation)
    props = propagators_for(system)
    H = hierarchy(gammas, interpretation)
    rows = []
    for obs, comps_list, max_weight in plan:
        exps = expansions(system, obs, max_weight)
        tags = all_tags(max_weight)
        for comps in _component_tuples(obs, comps_list):
            got = package_by_tag(system, props, exps, comps, method, kw,
                                 max_weight)
            rows += _rows(obs, comps, tags, got, H.moments(comps, tags, T))
    return rows


def compare_two_time(interpretation, gammas, method, kw, plan):
    """``⟨φ_a(t_min + T1) φ_b(t_min + T)⟩``: distinct times at one point."""
    system = make_system(gammas, interpretation)
    props = propagators_for(system)
    H = hierarchy(gammas, interpretation)
    rows = []
    for obs, comps_list, max_weight in plan:
        exps = expansions(system, obs, max_weight)
        tags = all_tags(max_weight)
        for comps in _component_tuples(obs, comps_list):
            got = package_by_tag(
                system, props, exps, comps, method, kw, max_weight,
                external_times={"x": T_MIN + T1, "y": T_MIN + T})
            ref = H.two_time(comps[0], comps[1], tags, T1, T)
            rows += _rows(("phi_a(x, t1)", "phi_b(y, t2)"), comps, tags, got,
                          ref)
    return rows


def _rows(obs, comps, tags, got, ref):
    rows = []
    for tag in tags:
        p, h = float(got.get(tag, 0.0)), ref[tag]
        rows.append(dict(observable=" ".join(obs), comps=list(comps),
                         tag=list(tag), package=p, hierarchy=h,
                         rel=abs(p - h) / abs(h) if h else abs(p)))
    return rows


#: ``(label, R, method, kwargs, [(observable, component tuples or None for
#: all, max weight)])``.  A tag of weight w collects vertex orders up to w,
#: so max weight is also the largest vertex order compared.  ``⟨φ_a φ_b⟩``
#: vanishes at odd weight and ``⟨φ_a⟩``, ``⟨φ_a φ_b φ_c⟩`` at even weight
#: (the Wick contraction needs an even number of unpaired φ's).
RUNS = [
    ("scalar R, gauss_legendre", "scalar", "gauss_legendre",
     dict(n_gauss=12),
     [(ONE, None, 3), (TWO, None, 3), (TWO, [(0, 1), (1, 1)], 4),
      (THREE, [(0, 1, 1), (1, 1, 0)], 3)]),
    ("scalar R, qmc_vectorized", "scalar", "qmc_vectorized",
     dict(n_samples=2 ** 14, seed=2), [(ONE, None, 2), (TWO, None, 2)]),
    ("matrix R, qmc_scalar", "matrix", "qmc_scalar",
     dict(n_samples=2 ** 12, seed=2), [(ONE, None, 3), (TWO, None, 2)]),
    ("matrix R, nquad", "matrix", "nquad", {},
     [(ONE, None, 1), (TWO, None, 2)]),
]

#: The two-time part: ``nquad`` and QMC converge, Gauss-Legendre does not
#: (see the module docstring).
TWO_TIME_RUNS = [
    ("nquad", {}), ("qmc_vectorized", dict(n_samples=2 ** 16, seed=3)),
    ("gauss_legendre", dict(n_gauss=12)),
]
TWO_TIME_PLAN = [(TWO, [(0, 1), (1, 1)], 3)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="white_l1_multiplicative_results.json")
    args = ap.parse_args()
    out = {}
    for interpretation in INTERPRETATIONS:
        for label, r_type, method, kw, plan in RUNS:
            t0 = time.perf_counter()
            rows = compare(interpretation, GAMMAS[r_type], method, kw, plan)
            secs = time.perf_counter() - t0
            _report(f"{interpretation}, {label}", rows, secs)
            out[f"{interpretation}, {label}"] = dict(rows=rows,
                                                     seconds=round(secs, 1))
        for method, kw in TWO_TIME_RUNS:
            t0 = time.perf_counter()
            rows = compare_two_time(interpretation, GAMMAS["scalar"], method,
                                    kw, TWO_TIME_PLAN)
            secs = time.perf_counter() - t0
            key = f"{interpretation}, scalar R, {method}, two times"
            _report(key, rows, secs)
            out[key] = dict(rows=rows, seconds=round(secs, 1))
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {args.out}")


def _report(key, rows, secs):
    worst: dict = {}
    for r in rows:
        w = (r["observable"], sum(r["tag"]))
        worst[w] = max(worst.get(w, 0.0), r["rel"])
    print(f"\n{key} ({secs:.0f} s, {len(rows)} comparisons)")
    for obs, w in sorted(worst):
        print(f"  {obs:34s} weight {w}: worst rel. difference "
              f"{worst[(obs, w)]:.1e}")


if __name__ == "__main__":
    main()
