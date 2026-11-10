"""The committed reference values of demos 4, 5 and 7 as ModelSpec cases.

Each demo checks the package against a hand-written moment hierarchy and
stores the hierarchy's value in a results file.  This module rebuilds every
such value from a :class:`~validation.modelspec.ModelSpec`: it states the
demo's model with the parameters copied from the demo's scripts, reads the
results file, and returns one :class:`Case` per stored value with the tags
over ``model.channels()`` whose coefficients add up to it.  Imports no
sft-wick code and none of the demos' modules.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .modelspec import (Kernel, ModelSpec, Multiplicative, Observable,
                        OUBlock, Poisson)

__all__ = ["Case", "demo4_cases", "demo5_cases", "demo5c_cases",
           "demo7_cases", "demo7_shot_cases", "all_cases", "unique_queries"]

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@dataclass(frozen=True)
class Case:
    """One stored reference value: the sum of ``moments(model, observable,
    tags)`` over ``tags`` should equal ``expected``.  ``group`` names the
    demo part (``'demo4'``, ``'demo5 part A'``, ``'demo7 item a'``, …)."""

    group: str
    source: str
    model: ModelSpec
    observable: Observable
    tags: tuple
    expected: float


def _load(relative: str) -> dict:
    with open(EXAMPLES / relative) as fh:
        return json.load(fh)


def _diagonal_ou(n: int, lam, sigma_t, kernels) -> tuple:
    """One OU block per component, loaded onto that component only."""
    lam = np.broadcast_to(lam, (n,))
    sigma_t = np.broadcast_to(sigma_t, (n,))
    if isinstance(kernels, Kernel):
        kernels = [kernels] * n
    return tuple(OUBlock(float(lam[a]), float(sigma_t[a]), kernels[a],
                         tuple(np.eye(n)[a])) for a in range(n))


def _legs(observable: str, comps=None) -> list:
    """``[(comp, label)]`` from ``'<phi_0(x) phi_1(y)>'`` or from
    ``'phi_a(x) phi_b(y)'`` with ``comps``."""
    found = re.findall(r"phi_(\w)\((\w)", observable)
    if comps is None:
        return [(int(c), lab) for c, lab in found]
    return [(int(c), lab) for c, (_, lab) in zip(comps, found)]


def _channel(text: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"-?\d+", text))


# ---------------------------------------------------------------------------
# Demo 4: two-component compound-Poisson noise (examples/demo4/)
# ---------------------------------------------------------------------------

#: ``poisson_noise.F_TENSOR``.
F_DEMO4 = np.array([[[0.25, -0.40], [0.15, 0.30]],
                    [[-0.35, 0.20], [0.45, -0.10]]])
T_DEMO4 = 1.7
POS_DEMO4 = {"x": 0.0, "y": 0.8, "z": -0.5, "w": 1.3}


def demo4_model(pulse: str, with_f: bool = True) -> ModelSpec:
    """``poisson_noise.Params``: ν = 1.5, h = (1, −0.7), s = (0.6, 1.1),
    τ = (0.3, 2), γ = 1; cumulant orders 3 and 4 kept."""
    return ModelSpec(
        gamma=(1.0, 1.0), quadratic=F_DEMO4 if with_f else None,
        poisson=Poisson(nu=1.5, amplitude=(1.0, -0.7), width=(0.6, 1.1),
                        pulse=pulse,
                        tau=(0.3, 2.0) if pulse == "exponential" else None,
                        orders=(3, 4)))


def _demo4_tags(k: int, j: int) -> tuple:
    """Tags ``(F, P3, P4)`` of the demo's ``(k_F, Σ(m − 2))``."""
    return tuple((k, p3, p4) for p3 in range(j + 1) for p4 in range(j + 1)
                 if p3 + 2 * p4 == j)


def demo4_cases() -> list:
    out = []
    for pulse in ("white", "exponential"):
        model = demo4_model(pulse)
        rows = _load("demo4/level_b_results.json")[pulse]["rows"]
        order4 = _load("demo4/level_b_order4_results.json")[pulse]["rows"]
        tagged = ([(r, tuple(r["tag"])) for r in rows]
                  + [(r, (3, 1)) for r in order4])
        for n, (row, (k, j)) in enumerate(tagged):
            obs = Observable(_legs(row["obs"]), POS_DEMO4, T_DEMO4)
            out.append(Case("demo4",
                            f"demo4 {pulse} {row['obs']} ({k}, {j}) #{n}",
                            model, obs, _demo4_tags(k, j),
                            row["hierarchy"]))
    return out


# ---------------------------------------------------------------------------
# Demo 5, part A: white noise (examples/demo5/white_model.py, run.py)
# ---------------------------------------------------------------------------

#: ``white_model.F_TENSOR`` (also demo 5 part C).
F_DEMO5 = np.array([[[0.30, -0.25], [0.10, 0.20]],
                    [[-0.15, 0.35], [0.40, -0.20]]])
S_DEMO5 = np.array([[0.50, 0.20], [0.20, 0.35]])


def demo5_model(variant: str) -> ModelSpec:
    S = {"mixing": S_DEMO5, "diagonal": np.diag(np.diag(S_DEMO5)),
         "isotropic": float(np.mean(np.diag(S_DEMO5))) * np.eye(2)}[variant]
    return ModelSpec(gamma=(1.0, 1.0), quadratic=F_DEMO5,
                     ou=_diagonal_ou(2, 0.4, 0.6, Kernel("gaussian", (0.9,))),
                     white=S, t_min=0.5)


def demo5_cases() -> list:
    out = []
    data = _load("demo5/results.json")
    pos = {"x": 0.0, "y": 0.7, "z": -0.4}
    for variant in ("mixing", "diagonal", "isotropic"):
        model = demo5_model(variant)
        for n, row in enumerate(data[variant]["rows"]):
            obs = Observable(_legs(row["observable"]), pos, 0.5 + 1.6)
            out.append(Case("demo5 part A",
                            f"demo5 {variant} {row['observable']} "
                            f"order {row['order']} #{n}", model, obs,
                            ((row["order"],),), row["hierarchy"]))
    return out


# ---------------------------------------------------------------------------
# Demo 5, part C: multiplicative white noise (white_l1_multiplicative.py)
# ---------------------------------------------------------------------------

G0_DEMO5C = np.array([[0.55, -0.20, 0.10], [0.15, 0.40, -0.25]])
G1_DEMO5C = np.array([[[0.20, -0.15], [-0.10, 0.25], [0.05, 0.10]],
                      [[-0.12, 0.18], [0.22, -0.08], [-0.15, 0.06]]])


def demo5c_model(gammas: tuple, interpretation: str) -> ModelSpec:
    return ModelSpec(gamma=gammas, quadratic=F_DEMO5,
                     ou=_diagonal_ou(2, 0.3, 0.7, Kernel("gaussian", (0.9,))),
                     multiplicative=Multiplicative(G0_DEMO5C, G1_DEMO5C,
                                                   interpretation),
                     t_min=0.4)


def demo5c_cases() -> list:
    out = []
    t_min, T, T1 = 0.4, 1.5, 0.9
    pos = {"x": 0.0, "y": 0.0, "z": 0.0}
    for key, block in _load("demo5/white_l1_multiplicative_results.json"
                            ).items():
        interpretation = key.split(",")[0]
        gammas = (1.1, 1.1) if "scalar R" in key else (0.7, 1.6)
        model = demo5c_model(gammas, interpretation)
        times = ({"x": t_min + T1, "y": t_min + T} if "two times" in key
                 else {})
        for n, row in enumerate(block["rows"]):
            obs = Observable(_legs(row["observable"], row["comps"]), pos,
                             t_min + T, times=times)
            out.append(Case("demo5 part C",
                            f"demo5c {key} {row['observable']} "
                            f"{row['comps']} {row['tag']} #{n}", model, obs,
                            (tuple(row["tag"]),), row["hierarchy"]))
    return out


# ---------------------------------------------------------------------------
# Demo 7: space, angle and time (space7_model.py, space7_run.py)
# ---------------------------------------------------------------------------

F_DEMO7 = np.array([[[0.30, -0.45], [0.20, 0.15]],
                    [[-0.25, 0.40], [0.35, -0.20]]])
G_DEMO7 = np.array([[[[0.12, -0.21], [0.17, 0.09]],
                     [[-0.14, 0.23], [0.06, -0.18]]],
                    [[[-0.11, 0.19], [0.08, -0.16]],
                     [[0.22, -0.07], [-0.13, 0.15]]]])
S_MIX_DEMO7 = np.array([[0.30, 0.12], [0.12, 0.22]])
T_DEMO7 = 1.3
POS_DEMO7 = {"x": 0.0, "y": 1.3, "z": -0.6}


def demo7_model(config: str) -> ModelSpec:
    """``space7_model.CONFIGS[config]``."""
    matrix = "matrixR" in config
    gamma = (0.7, 1.4) if matrix else (1.0, 1.0)
    white = S_MIX_DEMO7 if "white" in config else None
    if config.startswith("general"):
        kernels = [Kernel("inhomogeneous", (s, c, L))
                   for s, c, L in zip((0.9, 1.4), (0.8, -0.4), (0.9, 1.3))]
        ou = _diagonal_ou(2, (0.5, 0.35), (0.6, 1.1), kernels)
    else:
        if config.startswith("rotation"):
            kernel = Kernel("legendre", (0.25, 0.6, 0.3, 0.15))
        elif "gauss" in config:
            kernel = Kernel("gaussian", (0.8,))
        elif "custom" in config:
            kernel = Kernel("damped_cosine", (1.5, 1.6))
        else:
            kernel = Kernel("exponential", (0.9,))
        ou = _diagonal_ou(2, 0.5, 0.6, kernel)
    return ModelSpec(gamma=gamma, quadratic=F_DEMO7, cubic=G_DEMO7, ou=ou,
                     white=white, t_min=0.4)


def _direction(theta_deg: float, azimuth: float = 0.7) -> tuple:
    th = math.radians(theta_deg)
    return (math.sin(th) * math.cos(azimuth), math.sin(th) * math.sin(azimuth),
            math.cos(th))


def _demo7_observable(row: dict, t_min: float) -> Observable:
    legs = _legs(row["observable"], row["comps"])
    t_final = t_min + T_DEMO7
    if row["item"] == "a":
        tx, ty = row["times"]
        return Observable(legs, POS_DEMO7, t_min + max(tx, ty),
                          times={"x": t_min + tx, "y": t_min + ty})
    if row["item"] == "b":
        pos = {"x": (0.0, 0.0, 1.0), "y": _direction(row["angle"])}
        return Observable(legs, pos, t_final)
    over = row.get("integrate_over_", ())
    return Observable(legs, POS_DEMO7, t_final, integrate_over=over)


def demo7_cases() -> list:
    out = []
    models = {}
    for n, row in enumerate(_load("demo7/results.json")["rows"]):
        if row["status"] != "ok":
            continue
        model = models.setdefault(row["config"], demo7_model(row["config"]))
        obs = _demo7_observable(row, model.t_min)
        for tag, (_, reference) in row["channels"].items():
            out.append(Case(f"demo7 item {row['item']}",
                            f"demo7 item {row['item']} {row['config']} "
                            f"{row['observable']} {row['comps']} {tag} #{n}",
                            model, obs, (_channel(tag),), reference))
    return out


# ---------------------------------------------------------------------------
# Demo 7, item (d): compound-Poisson noise in three dimensions
# ---------------------------------------------------------------------------

F_SHOT = np.array([[[0.20, -0.35], [0.25, 0.10]],
                   [[-0.30, 0.15], [0.40, -0.25]]])
POINTS_SHOT = {"x": (0.0, 0.0, 0.0), "y": (0.5, -0.3, 0.4),
               "z": (-0.2, 0.6, -0.35)}


def demo7_shot_model(variant: str) -> ModelSpec:
    """``space7_shot.ShotParams``; cumulant order 3 kept."""
    pulse = "exponential" if variant == "exponential" else "white"
    return ModelSpec(
        gamma=(0.8, 1.3) if variant == "white-matrixR" else (1.0, 1.0),
        quadratic=F_SHOT,
        poisson=Poisson(nu=1.2, amplitude=(0.9, -0.6), width=(0.5, 0.9),
                        pulse=pulse,
                        tau=(0.35, 1.8) if pulse == "exponential" else None,
                        orders=(3,)),
        t_min=0.3)


def demo7_shot_cases() -> list:
    out = []
    models = {}
    for n, row in enumerate(_load("demo7/shot3d_results.json")["rows"]):
        if row["status"] != "ok":
            continue
        model = models.setdefault(row["variant"],
                                  demo7_shot_model(row["variant"]))
        obs = Observable(_legs(row["observable"], row["comps"]), POINTS_SHOT,
                         model.t_min + 1.4)
        for tag, (_, reference) in row["channels"].items():
            out.append(Case("demo7 item d",
                            f"demo7 shot3d {row['variant']} "
                            f"{row['observable']} {row['comps']} {tag} #{n}",
                            model, obs, (_channel(tag),), reference))
    return out


def all_cases() -> list:
    """Every case of every demo, in the order of the functions above."""
    return (demo4_cases() + demo5_cases() + demo5c_cases() + demo7_cases()
            + demo7_shot_cases())


def unique_queries(cases) -> dict:
    """``{(model, observable): sorted tags}``: one hierarchy solve each."""
    out: dict = {}
    for case in cases:
        out.setdefault((case.model, case.observable), set()).update(case.tags)
    return {key: sorted(tags) for key, tags in out.items()}

