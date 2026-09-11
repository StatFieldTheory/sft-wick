r"""Demo 5, part B: multiplicative white noise, L0, against the Itô hierarchy.

.. math::

    dφ_a = (−γ_a φ_a + F_{abc} φ_b φ_c)\,dt + G_{ad}(φ)\,dW_d ,
    \qquad G_{ad}(φ) = g_{ad} + g'_{adc} φ_c ,

Itô convention, one point, ``φ(0) = 0``.  In the MSR action the noise gives
``½ ψ_a ψ_b S_ab(φ)`` with ``S = G Gᵀ``:

* ``S⁰_ab = g_ad g_bd``: white noise, in C;
* ``S¹_abc = g_ad g'_bdc + g'_adc g_bd``: a local ψψφ vertex;
* ``S²_abce = g'_adc g'_bde``: a local ψψφφ vertex.

A vertex with two ψ legs takes the factor ``−i²/2! = ½`` (checked here by
making an order-1 ψψ vertex reproduce C).  The L1 workflow cannot build a
local vertex with two ψ legs, so this part is L0: ``compute_moment`` and
``integrate_diagrams``.

With distinct decay rates ``γ_a`` R is a (diagonal) matrix, and a ψψφ vertex
whose two ψ legs contract with the two φ legs of one F vertex puts two R
propagators between the same two points: the diagrams the repeated-pair fix
(63fc842, merged 2026-09-11) is about.  Matrix R runs only on the
scalar-loop backends; with equal rates R is a scalar and Gauss-Legendre
runs too.

Tags ``(k, j)`` count powers of F and of ``g'``.  Run
``python multiplicative.py``; results go to ``multiplicative_results.json``.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("SFT_WICK_QUIET_CACHE", "1")

import sft_wick as sw  # noqa: E402
from sft_wick.evaluate import integrate_diagrams  # noqa: E402
from sft_wick.workflow.specs import ConstantImpulse  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import ito_moments as im  # noqa: E402

N = 2
T = 1.5
F = np.array([[[0.30, -0.25], [0.10, 0.20]],
              [[-0.15, 0.35], [0.40, -0.20]]])
G0 = np.array([[0.60, 0.15], [-0.10, 0.45]])
G1 = np.array([[[0.20, -0.10], [0.05, 0.15]],
               [[-0.12, 0.08], [0.18, -0.06]]])        # g'_{adc}
S0 = G0 @ G0.T
S1 = (np.einsum("ad,bdc->abc", G0, G1) + np.einsum("adc,bd->abc", G1, G0))
S2 = np.einsum("adc,bde->abce", G1, G1)

#: Channels: (label, observable, order, vertex set, tag).
CHANNELS = [
    ("C", ("phi_a(x)", "phi_b(y)"), 0, {}, (0, 0)),
    ("H", ("phi_a(x)", "phi_b(y)"), 1, {"H"}, (0, 2)),
    ("FG", ("phi_a(x)", "phi_b(y)"), 2, {"F", "G"}, (1, 1)),
    ("GG", ("phi_a(x)", "phi_b(y)"), 2, {"G"}, (0, 2)),
    ("F tadpole", ("phi_a(x)",), 1, {"F"}, (1, 0)),
    ("FH", ("phi_a(x)",), 2, {"F", "H"}, (1, 2)),
    ("FG", ("phi_a(x)",), 2, {"F", "G"}, (1, 1)),
]


def _cache(gammas):
    """The exact C of the white noise ``S⁰`` (built-in closed form)."""
    system = sw.System(
        field=sw.FieldSpec("phi", N), linear=sw.DiagonalA(gamma=list(gammas)),
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.0, sigma_t=1.0),
                spatial=sw.ExponentialSpatial(sigma_x=1.0)),
            sigma2=ConstantImpulse(S0)))
    return system.propagators(t_max=T + 0.5, c_closed_form="auto",
                              c_closed_form_only=True,
                              c_closed_form_vectorized=True, diag_C=False,
                              progress=False).cache


def _terms(obs, order, vertices, iso_R):
    sw.reset_uid_counter()
    phi = sw.Field("phi", "physical", n_components=N)
    psi = sw.Field("psi", "response", n_components=N)
    fields = {"F": [psi, phi, phi], "G": [psi, psi, phi],
              "H": [psi, psi, phi, phi]}
    action = sw.Action(vertices=[sw.Vertex(fields=fields[v], coupling=v,
                                           local=True)
                                 for v in sorted(vertices)])
    ops = [phi(c, lab) for c, lab in
           zip("ab", [o[o.index("(") + 1:-1] for o in obs])]
    res = sw.compute_moment(ops, action, order=order, iso_R=iso_R,
                            diag_R=True, diag_C=False)
    names = set(vertices)
    # keep the diagrams that use every vertex of the channel
    return [dt for dt in res.diagram_terms(order)
            if names == _symbols(dt)]


def _symbols(dt):
    from sft_wick.workflow.expansion import _collect_symbol_names
    return set(_collect_symbol_names(dt.coupling_sum))


COUPLINGS = {"F": -1j * F, "G": 0.5 * S1, "H": 0.5 * S2}


def package(gammas, method, max_order=2, **kw):
    iso_R = len(set(gammas)) == 1
    cache = _cache(gammas)
    rows = []
    for label, obs, order, vertices, tag in CHANNELS:
        if order > max_order:
            continue
        dts = (_terms(obs, order, vertices, iso_R) if order
               else _terms(obs, 0, set(), iso_R))
        for comps in itertools.product(range(N), repeat=len(obs)):
            fixed = dict(zip("ab", comps))
            if not dts:
                total = 0.0
            else:
                total, _ = integrate_diagrams(
                    dts, COUPLINGS, lambda_f=T, cache=cache, method=method,
                    fixed_indices=fixed,
                    positions={"x": 0.0, "y": 0.0}, **kw)
            rows.append(dict(channel=label, order=order, tag=list(tag),
                             comps=list(comps), package=float(total),
                             n_terms=len(dts)))
    return rows


def compare(gammas, rows):
    """Sum the package's channels per (tag, component tuple) and compare
    with the hierarchy's coefficient of that tag: under Itô the GG channel
    has no diagram, so the ``g'²`` coefficient of ``⟨φ φ⟩`` is H alone."""
    groups: dict = {}
    for r in rows:
        key = (tuple(r["tag"]), tuple(r["comps"]))
        g = groups.setdefault(key, dict(channels=[], package=0.0,
                                        n_terms=0))
        g["channels"].append(r["channel"])
        g["package"] += r["package"]
        g["n_terms"] += r["n_terms"]
    out = [dict(tag=list(tag), comps=list(comps),
                channels="+".join(g["channels"]), n_terms=g["n_terms"],
                package=g["package"])
           for (tag, comps), g in groups.items()]
    return hierarchy(gammas, out)


def hierarchy(gammas, rows):
    sde = im.PolySDE(D=N, n_tags=2)
    sde.add_linear_drift(-np.diag(gammas))
    sde.add_quadratic_drift(F, tag=(1, 0))
    for a, b in itertools.product(range(N), repeat=2):
        sde.diffusion.append((a, b, S0[a, b], (0,) * N, (0, 0)))
        for c in range(N):
            sde.diffusion.append((a, b, S1[a, b, c], im.unit(N, c), (0, 1)))
            for e in range(N):
                sde.diffusion.append((a, b, S2[a, b, c, e],
                                      im.unit(N, c, e), (0, 2)))
    for r in rows:
        mono = im.unit(N, *r["comps"])
        tag = tuple(r["tag"])
        r["hierarchy"] = float(im.solve(sde, [(mono, tag)], [T])[(mono, tag)][0])
        r["rel"] = (abs(r["package"] - r["hierarchy"]) / abs(r["hierarchy"])
                    if r["hierarchy"] else abs(r["package"]))
    return rows


def main():
    out = {}
    for label, gammas, method, kw in [
        ("scalar R, gauss_legendre", (1.0, 1.0), "gauss_legendre",
         dict(n_gauss=16)),
        ("matrix R, nquad (order <= 1)", (0.6, 1.6), "nquad",
         dict(max_order=1)),
        ("matrix R, qmc_scalar", (0.6, 1.6), "qmc_scalar",
         dict(n_samples=2 ** 13, seed=3)),
    ]:
        t0 = time.perf_counter()
        rows = compare(np.array(gammas), package(gammas, method, **kw))
        secs = time.perf_counter() - t0
        print(f"\n{label} ({secs:.0f} s)")
        print(f"  {'tag (F, g1)':>11} {'comps':>8} {'channels':>10} "
              f"{'terms':>5} {'package':>16} {'hierarchy':>16} {'rel':>8}")
        for r in rows:
            print(f"  {str(tuple(r['tag'])):>11} {str(tuple(r['comps'])):>8} "
                  f"{r['channels']:>10} {r['n_terms']:>5} "
                  f"{r['package']:+16.9e} {r['hierarchy']:+16.9e} "
                  f"{r['rel']:8.1e}")
        out[label] = dict(rows=rows, seconds=round(secs, 1))
    with open("multiplicative_results.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote multiplicative_results.json")


if __name__ == "__main__":
    main()
