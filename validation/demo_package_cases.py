"""The committed *package* values of demos 4, 5 and 7 as lowering cases.

:mod:`validation.demo_cases` rebuilds the demos' hierarchy values; this
module states, for every committed package value whose route the lowering
takes too, the ModelSpec, observable, integrator settings, C source and
flags that produced it, so that ``validation/lower.py`` can reproduce it.
The route is the same when C is exact on both sides (the package's
built-in closed form for demos 5 and 7, the demos' own closed forms for
demo 4 and item (d) against the lowering's) and the vertices are lowered
alike.  Two routes differ and are left out: the exponential pulses of demo
4 and of item (d), which the demos lower as ``already_R_contracted`` K_R
and the lowering as the raw cumulant, and the quadrature-table items (b),
(c) of demo 7, which tabulate other specs.  Imports no sft-wick code.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import demo_cases as dc
from .modelspec import Observable

__all__ = ["PackageCase", "package_cases"]


@dataclass(frozen=True)
class PackageCase:
    """A committed package value: the sum over ``tags`` of the lowering's
    values for one evaluation should equal ``expected``."""

    group: str
    source: str
    model: object
    observable: Observable
    max_weight: int
    orders: tuple | None
    method: str
    settings: tuple
    c_source: str
    flags: tuple
    tags: tuple
    expected: float

    def job(self) -> dict:
        """The evaluation, as ``validation.lower.run_job`` takes it (less
        the JSON of the model and observable)."""
        return dict(max_weight=self.max_weight,
                    orders=None if self.orders is None else list(self.orders),
                    method=self.method, settings=dict(self.settings),
                    c_source=self.c_source, flags=dict(self.flags))


def _items(d: dict) -> tuple:
    return tuple(sorted(d.items()))


# -- demo 4 (white pulses: the raw equal_time vertex on both sides) ----------

_DEMO4_ORDER = {"order 0": 0, "FK3": 2, "FF": 2, "FFK4": 3, "F tadpole": 1}
_DEMO4_TAG = {"order 0": (0, 0, 0), "FK3": (1, 1, 0), "FF": (2, 0, 0),
              "FFK4": (2, 0, 1), "F tadpole": (1, 0, 0)}


def _demo4() -> list:
    out = []
    model = dc.demo4_model("white")
    rows = dc._load("demo4/level_b_results.json")["white"]["rows"]
    for n, row in enumerate(rows):
        order = _DEMO4_ORDER[row["channel"]]
        n_gauss = 32 if row["channel"] == "F tadpole" else {2: 16, 3: 16}.get(
            order, 8)
        obs = Observable(dc._legs(row["obs"]), dc.POS_DEMO4, dc.T_DEMO4)
        out.append(PackageCase(
            "demo4", f"demo4 white {row['obs']} {row['channel']} #{n}",
            model, obs, order, (order,), "gauss_legendre",
            _items(dict(n_gauss=n_gauss)), "oracle", _items(dict(diag_C=False)),
            (_DEMO4_TAG[row["channel"]],), row["package"]))
    for n, row in enumerate(dc._load("demo4/level_b_order4_results.json")
                            ["white"]["rows"]):
        obs = Observable(dc._legs(row["obs"]), dc.POS_DEMO4, dc.T_DEMO4)
        out.append(PackageCase(
            "demo4 order 4", f"demo4 white {row['obs']} F3K3 #{n}", model,
            obs, 4, (4,), "gauss_legendre", _items(dict(n_gauss=12)),
            "oracle", _items(dict(diag_C=False)), ((3, 1, 0),),
            row["package"]))
    return out


# -- demo 5, part A (the built-in closed form) -------------------------------

_DEMO5_SETTINGS = {"gauss_legendre": dict(n_gauss=12),
                   "qmc_vectorized": dict(n_samples=2 ** 14, seed=2),
                   "qmc_scalar": dict(n_samples=2 ** 11, seed=2),
                   "qmc": dict(n_samples=2 ** 11, seed=2), "nquad": {}}
_DEMO5_FLAGS = {"mixing": dict(diag_C=False), "diagonal": dict(diag_C=True),
                "isotropic": dict(diag_C=True, iso_C=True)}


def _demo5() -> list:
    out = []
    data = dc._load("demo5/results.json")
    pos = {"x": 0.0, "y": 0.7, "z": -0.4}
    for variant, flags in _DEMO5_FLAGS.items():
        model = dc.demo5_model(variant)
        for n, row in enumerate(data[variant]["rows"]):
            order = row["order"]
            obs = Observable(dc._legs(row["observable"]), pos, 0.5 + 1.6)
            out.append(PackageCase(
                "demo5 part A", f"demo5 {variant} {row['observable']} order "
                f"{order} {row['method']} #{n}", model, obs, order, (order,),
                row["method"], _items(_DEMO5_SETTINGS[row["method"]]), "auto",
                _items(flags), ((order,),), row["package"]))
    return out


# -- demo 5, part C (the built-in closed form, MultiplicativeImpulse) --------

_DEMO5C_SETTINGS = {
    "scalar R, gauss_legendre": ("gauss_legendre", dict(n_gauss=12)),
    "scalar R, qmc_vectorized": ("qmc_vectorized",
                                 dict(n_samples=2 ** 14, seed=2)),
    "matrix R, qmc_scalar": ("qmc_scalar", dict(n_samples=2 ** 12, seed=2)),
    "matrix R, nquad": ("nquad", {}),
    "scalar R, nquad, two times": ("nquad", {}),
    "scalar R, qmc_vectorized, two times": ("qmc_vectorized",
                                            dict(n_samples=2 ** 16, seed=3)),
    "scalar R, gauss_legendre, two times": ("gauss_legendre",
                                            dict(n_gauss=12)),
}


def _demo5c() -> list:
    out = []
    t_min, T, T1 = 0.4, 1.5, 0.9
    pos = {"x": 0.0, "y": 0.0, "z": 0.0}
    for key, block in dc._load("demo5/white_l1_multiplicative_results.json"
                               ).items():
        interpretation, label = key.split(", ", 1)
        method, settings = _DEMO5C_SETTINGS[label]
        gammas = (1.1, 1.1) if "scalar R" in key else (0.7, 1.6)
        model = dc.demo5c_model(gammas, interpretation)
        times = ({"x": t_min + T1, "y": t_min + T} if "two times" in key
                 else {})
        groups: dict = {}
        for row in block["rows"]:
            groups.setdefault((row["observable"], tuple(row["comps"])),
                              []).append(row)
        for (observable, comps), rows in groups.items():
            weight = max(sum(r["tag"]) for r in rows)
            obs = Observable(dc._legs(observable, comps), pos, t_min + T,
                             times=times)
            for n, row in enumerate(rows):
                out.append(PackageCase(
                    "demo5 part C", f"demo5c {key} {observable} {list(comps)}"
                    f" {row['tag']} #{n}", model, obs, weight, None, method,
                    _items(settings), "auto", _items(dict(diag_C=False)),
                    (tuple(row["tag"]),), row["package"]))
    return out


# -- demo 7: the closed-form configurations, and item (d) with white pulses --

_DEMO7_CLOSED = ("translation-exp", "translation-exp-matrixR",
                 "translation-exp-white-matrixR")


def _demo7() -> list:
    out = []
    for n, row in enumerate(dc._load("demo7/results.json")["rows"]):
        if row["status"] != "ok" or row["config"] not in _DEMO7_CLOSED:
            continue
        model = dc.demo7_model(row["config"])
        obs = dc._demo7_observable(row, model.t_min)
        flags = dict(diag_C=model.white is None)
        for tag, (package, _) in row["channels"].items():
            out.append(PackageCase(
                f"demo7 item {row['item']}", f"demo7 {row['config']} "
                f"{row['observable']} {row['comps']} {row['method']} {tag} "
                f"#{n}", model, obs, row["order"], (row["order"],),
                row["method"], _items(row["settings"]), "auto", _items(flags),
                (dc._channel(tag),), package))
    for n, row in enumerate(dc._load("demo7/shot3d_results.json")["rows"]):
        if row["status"] != "ok" or row["variant"] == "exponential":
            continue
        model = dc.demo7_shot_model(row["variant"])
        obs = Observable(dc._legs(row["observable"], row["comps"]),
                         dc.POINTS_SHOT, model.t_min + 1.4)
        for tag, (package, _) in row["channels"].items():
            out.append(PackageCase(
                "demo7 item d", f"demo7 shot3d {row['variant']} "
                f"{row['observable']} {row['comps']} {row['method']} {tag} "
                f"#{n}", model, obs, row["order"], (row["order"],),
                row["method"], _items(row["settings"]), "oracle",
                _items(dict(diag_C=False)), (dc._channel(tag),), package))
    return out


def package_cases() -> list:
    return _demo4() + _demo5() + _demo5c() + _demo7()
