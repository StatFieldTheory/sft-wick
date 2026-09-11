r"""Demo 7: observables in space, angle and time against the exact hierarchy.

Items (see ``README.md``):

(a) the two-time function ``⟨φ_a(x, t) φ_b(y, t')⟩``, orders 0-2, both
    time orders, through ``external_times`` on every integrator;
(b) rotation-invariant noise with a four-term Legendre kernel, three angles;
(c) the package's quadrature tables for a Gaussian and a custom
    translation kernel, and for a ``GeneralKappa2`` without spatial symmetry;
(e) ``integrate_over`` (``'all'`` and one point) at N = 2 with a ≠ b, and
    the three-point function at order 2.

Every package value is compared, channel by channel (``F``, ``G``, ``FG``),
with the moment hierarchy of the Markov embedding at the observation points
(:mod:`space7_reference`, no sft-wick code).  Item (d) is
``space7_shot3d.py``.

Run ``python space7_run.py`` (``--items a,c`` for a subset); results go to
``results.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("SFT_WICK_QUIET_CACHE", "1")

import numpy as np  # noqa: E402

import space7_model as m  # noqa: E402

#: Observation time and the earlier time of item (a), measured from t_min.
T = 1.3
T_EARLY = 0.65
POS = {"x": 0.0, "y": 1.3, "z": -0.6}
#: Polar angles of the second point from the first (item b), degrees.
ANGLES = (35.0, 80.0, 150.0)
AZIMUTH = 0.7
NORTH = np.array([0.0, 0.0, 1.0])
TAGS = {0: [(0, 0)], 1: [(1, 0), (0, 1)], 2: [(2, 0), (0, 2), (1, 1)]}
TWO = ("phi_a(x)", "phi_b(y)")
THREE = ("phi_a(x)", "phi_b(y)", "phi_c(z)")


def GL(n):
    return ("gauss_legendre", {"n_gauss": n})


def QV(k):
    return ("qmc_vectorized", {"n_samples": 2 ** k, "seed": 5})


def QS(k):
    return ("qmc_scalar", {"n_samples": 2 ** k, "seed": 5})


NQ = ("nquad", {})


def direction(theta_deg: float, azimuth: float = AZIMUTH) -> np.ndarray:
    """Unit vector at polar angle ``θ`` from :data:`NORTH`."""
    th = np.deg2rad(theta_deg)
    return np.array([np.sin(th) * np.cos(azimuth),
                     np.sin(th) * np.sin(azimuth), np.cos(th)])


class Setup:
    """A configuration's system, propagators and expansions."""

    def __init__(self, cfg: m.Config):
        self.cfg = cfg
        self.system = cfg.system()
        self.props = cfg.propagators(self.system, t_max=cfg.t_min + T + 0.3)
        self._exp: dict = {}

    def expansion(self, observable, orders=(0, 1, 2)):
        key = (observable, tuple(orders))
        if key not in self._exp:
            self._exp[key] = self.system.expand(
                observable, orders=list(orders), progress=False,
                **self.cfg.expand_flags())
        return self._exp[key]


def compare(result, ref: dict, order: int, channel_tag=m.channel_tag) -> dict:
    """The package's channels (``Result.by_vertex_type``) against the
    reference's tags.  ``error`` is the worst relative difference over the
    non-zero channels and the total; ``zero`` the largest package value
    where the reference is exactly 0."""
    pkg = {channel_tag(label, order): float(v)
           for label, v in result.by_vertex_type.items()}
    channels, err, zero = {}, 0.0, 0.0
    for tag in sorted(set(ref) | set(pkg)):
        p, r = pkg.get(tag, 0.0), float(ref.get(tag, 0.0))
        channels[str(tag)] = [p, r]
        if r != 0.0:
            err = max(err, abs(p - r) / abs(r))
        else:
            zero = max(zero, abs(p))
    total_p, total_r = float(result.total), float(sum(ref.values()))
    if total_r != 0.0:
        err = max(err, abs(total_p - total_r) / abs(total_r))
    return dict(package=total_p, reference=total_r, error=err, zero=zero,
                channels=channels)


def run_case(setup: Setup, observable, comps, order, route, *, ref,
             positions, t_final, orders=(0, 1, 2), external_times=None,
             integrate_over=None, **meta) -> dict:
    """One package evaluation against ``ref`` (``{tag: value}``)."""
    method, kw = route
    row = dict(config=setup.cfg.name, observable=" ".join(observable),
               comps=list(comps), order=order, method=method,
               settings=dict(kw), **meta)
    t0 = time.perf_counter()
    try:
        res = setup.expansion(observable, orders).evaluate(
            setup.props, positions=positions, t_final=t_final,
            component_pair=tuple(comps), orders=[order], method=method,
            external_times=external_times, integrate_over=integrate_over,
            **kw)
    except NotImplementedError as exc:
        row.update(status="refused", reason=str(exc).split(".")[0])
        return row
    row.update(status="ok", seconds=round(time.perf_counter() - t0, 2),
               **compare(res, ref, order))
    return row


# ---------------------------------------------------------------------------
# (a) two-time function
# ---------------------------------------------------------------------------

PLAN_A = {
    "translation-exp": [GL(48), QV(16), QS(11), NQ],
    "translation-exp-matrixR": [GL(48), QV(16), QS(11), NQ],
    "translation-exp-white-matrixR": [GL(48), QV(16), QS(12), NQ],
}


def item_a(plan=PLAN_A, pairs=((0, 1), (1, 0), (1, 1)), orders=(0, 1, 2),
           time_orders=((T, T_EARLY), (T_EARLY, T))) -> list:
    rows = []
    for name, routes in plan.items():
        s = Setup(m.CONFIGS[name])
        H = s.cfg.hierarchy([POS["x"], POS["y"]])
        for tx, ty in time_orders:
            ext = {"x": s.cfg.t_min + tx, "y": s.cfg.t_min + ty}
            for ab in pairs:
                for order in orders:
                    if tx >= ty:
                        ref = H.two_time([(ab[0], 0)], [(ab[1], 1)],
                                         TAGS[order], tx, ty)
                    else:
                        ref = H.two_time([(ab[1], 1)], [(ab[0], 0)],
                                         TAGS[order], ty, tx)
                    for route in routes:
                        rows.append(run_case(
                            s, TWO, ab, order, route, ref=ref, positions=POS,
                            t_final=s.cfg.t_min + max(tx, ty),
                            external_times=ext, item="a",
                            times=[tx, ty]))
    return rows


# ---------------------------------------------------------------------------
# (b) rotation-invariant noise, Legendre kernel with four coefficients
# ---------------------------------------------------------------------------

PLAN_B = {
    "rotation-legendre": {0: [GL(32)], 1: [GL(32), QS(10)],
                          2: [GL(32), QV(14)]},
    "rotation-legendre-matrixR": {0: [QS(11)], 1: [QS(11), GL(32)],
                                  2: [QS(11), QV(14)]},
}


def item_b(plan=PLAN_B, angles=ANGLES, pairs=((0, 1), (1, 1)),
           orders=(0, 1, 2)) -> list:
    rows = []
    for name, routes in plan.items():
        s = Setup(m.CONFIGS[name])
        for theta in angles:
            y = direction(theta)
            H = s.cfg.hierarchy([NORTH, y])
            for ab in pairs:
                for order in orders:
                    ref = H.moments([(ab[0], 0), (ab[1], 1)], TAGS[order], T)
                    for route in routes[order]:
                        rows.append(run_case(
                            s, TWO, ab, order, route, ref=ref,
                            positions={"x": NORTH, "y": y},
                            t_final=s.cfg.t_min + T, item="b", angle=theta))
    return rows


# ---------------------------------------------------------------------------
# (c) quadrature tables: Gaussian, custom and general kernels
# ---------------------------------------------------------------------------

PLAN_C = {
    "translation-gauss-quad": {0: [GL(32)], 1: [GL(32)],
                               2: [GL(32), QV(14)]},
    "translation-custom-quad": {0: [GL(32)], 1: [GL(32)],
                                2: [GL(32), QV(14)]},
    "general-quad": {0: [GL(32)], 1: [GL(32)], 2: [GL(32), QV(14)]},
    "general-quad-matrixR": {0: [QS(11)], 1: [QS(11), GL(32)],
                             2: [QS(11), QV(14)]},
}


def item_c(plan=PLAN_C, pairs=((0, 1), (1, 1)), orders=(0, 1, 2)) -> list:
    rows = []
    for name, routes in plan.items():
        cfg = m.CONFIGS[name]
        s = Setup(cfg)
        closed = (Setup(cfg.with_(c_source="closed_form"))
                  if cfg.homogeneity == "translation" else None)
        H = cfg.hierarchy([POS["x"], POS["y"]])
        for ab in pairs:
            for order in orders:
                ref = H.moments([(ab[0], 0), (ab[1], 1)], TAGS[order], T)
                for route in routes[order]:
                    row = run_case(s, TWO, ab, order, route, ref=ref,
                                   positions=POS, t_final=cfg.t_min + T,
                                   item="c")
                    if closed is not None and row["status"] == "ok":
                        cf = run_case(closed, TWO, ab, order, route, ref=ref,
                                      positions=POS, t_final=cfg.t_min + T)
                        row["closed_form_error"] = cf["error"]
                        row["table_vs_closed_form"] = (
                            abs(row["package"] - cf["package"])
                            / abs(cf["package"]) if cf["package"] else 0.0)
                    rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# (e) integrate_over and the three-point function
# ---------------------------------------------------------------------------

PLAN_E_OVER = {
    "translation-exp": [GL(24), QV(16), QS(12)],
    "translation-exp-matrixR": [QS(12), GL(24), QV(16)],
}
PLAN_E_THREE = {
    "translation-exp": [GL(24), QV(14)],
    "translation-exp-matrixR": [QS(11), GL(24)],
}


def item_e(plan_over=PLAN_E_OVER, plan_three=PLAN_E_THREE,
           pairs=((0, 1), (1, 0), (1, 1)), orders=(0, 1, 2),
           triples=((0, 1, 1), (1, 0, 0), (1, 1, 0))) -> list:
    rows = []
    for name, routes in plan_over.items():
        s = Setup(m.CONFIGS[name])
        for ab in pairs:
            H = s.cfg.hierarchy([POS["x"], POS["y"]],
                                integrated=[(ab[0], 0), (ab[1], 1)])
            for over, legs in [("all", [("I", ab[0], 0), ("I", ab[1], 1)]),
                               (("x",), [("I", ab[0], 0), (ab[1], 1)])]:
                for order in orders:
                    ref = H.moments(legs, TAGS[order], T)
                    for route in routes:
                        rows.append(run_case(
                            s, TWO, ab, order, route, ref=ref, positions=POS,
                            t_final=s.cfg.t_min + T,
                            integrate_over=(over if over == "all"
                                            else set(over)),
                            item="e", integrate_over_=(
                                over if over == "all" else list(over))))
    for name, routes in plan_three.items():
        s = Setup(m.CONFIGS[name])
        H = s.cfg.hierarchy([POS["x"], POS["y"], POS["z"]])
        for abc in triples:
            for order in (1, 2):
                ref = H.moments([(c, i) for i, c in enumerate(abc)],
                                TAGS[order], T)
                for route in routes:
                    rows.append(run_case(
                        s, THREE, abc, order, route, ref=ref, positions=POS,
                        t_final=s.cfg.t_min + T, orders=(1, 2), item="e"))
    return rows


# ---------------------------------------------------------------------------

ITEMS = {"a": item_a, "b": item_b, "c": item_c, "e": item_e}


def summarise(rows) -> dict:
    """Worst error per (item, config, method, order) over the ok rows, and
    the refused routes."""
    out: dict = {}
    for r in rows:
        key = f"{r['item']} | {r['config']} | {r['method']} | order {r['order']}"
        if r.get("integrate_over_") is not None:
            key += f" | integrate_over={r['integrate_over_']}"
        if r["observable"].count("phi") == 3:
            key += " | 3-point"
        e = out.setdefault(key, {"worst": 0.0, "zero": 0.0, "n": 0,
                                 "refused": 0})
        if r["status"] == "ok":
            e["worst"] = max(e["worst"], r["error"])
            e["zero"] = max(e["zero"], r["zero"])
            e["n"] += 1
        else:
            e["refused"] += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", default="a,b,c,e")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()
    rows = []
    timing = {}
    for item in args.items.split(","):
        t0 = time.perf_counter()
        rows += ITEMS[item]()
        timing[item] = round(time.perf_counter() - t0, 1)
        print(f"item ({item}): {timing[item]} s")
    summary = summarise(rows)
    for key, e in summary.items():
        worst = f"{e['worst']:.1e}" if e["n"] else "-"
        print(f"  {key:78s} worst {worst:>8s}  zero {e['zero']:.1e}  "
              f"n {e['n']:3d}  refused {e['refused']}")
    with open(args.out, "w") as fh:
        json.dump(dict(summary=summary, seconds=timing, rows=rows), fh,
                  indent=1, default=float)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
