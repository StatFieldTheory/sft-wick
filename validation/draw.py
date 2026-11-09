"""Random models and observables for the fuzzer, drawn from a fixed seed.

:func:`draw` turns a seed into a :class:`Case`: a ModelSpec, an observable,
the largest tag weight compared, and the expansion flags the model admits.
Three kinds of model are drawn.  A ``general`` model takes every feature
independently: rates, widths and loadings differ between components (in a
tenth of them the rates are equal, a scalar R beside a dense C), amplitudes
take both signs, points are distinct, legs are labelled in a random order,
and ``t_min`` is not 0.  Its observables with two or more legs pin one
leg's time with probability 0.6.  A ``diagonal`` model has a diagonal C, so
``diag_C`` applies.  An ``isotropic`` model has equal rates and ``C = c·I``,
so ``iso_R`` and ``iso_C`` apply as well.  Exponential Poisson pulses keep
cumulant orders up to 3, white pulses up to 4: order 4 with exponential
pulses costs minutes per integrator.

The same seed gives the same case on every platform and numpy version: only
``numpy.random.default_rng`` draws, in a fixed order, and every derived
number is formed in Python floats (a BLAS product rounds differently from
build to build).  Imports no sft-wick code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .modelspec import (Cumulant, Kernel, Leg, ModelSpec, Multiplicative,
                        Observable, OUBlock, Poisson, as_array)

__all__ = ["Case", "draw", "admissible_flags", "KINDS"]

#: Kinds of model and the probability of each.
KINDS = {"general": 0.6, "diagonal": 0.2, "isotropic": 0.2}
LABELS = "xyz"


@dataclass(frozen=True)
class Case:
    """A drawn model and observable: compare tags of weight up to
    ``max_weight``, with ``flags`` passed to the expansion."""

    seed: int
    kind: str
    model: ModelSpec
    observable: Observable
    max_weight: int
    flags: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

def _white_total(model: ModelSpec) -> np.ndarray:
    N = model.n_components
    S = np.zeros((N, N))
    if model.white is not None:
        S += as_array(model.white)
    if model.multiplicative is not None:
        g0 = as_array(model.multiplicative.g0)
        S += g0 @ g0.T
    return S


def _diagonal(model: ModelSpec) -> bool:
    if model.poisson is not None and model.n_components > 1:
        return False
    if np.any(np.triu(_white_total(model), 1) != 0.0):
        return False
    return all(np.count_nonzero(blk.loading) <= 1 for blk in model.ou)


def _isotropic(model: ModelSpec) -> bool:
    """``C = c·I``: equal rates, and every component sees the same noise."""
    N = model.n_components
    if N == 1:
        return True
    if not _diagonal(model) or len(set(model.gamma)) != 1:
        return False
    if len(set(np.diag(_white_total(model)))) != 1:
        return False
    seen = []
    for blk in model.ou:
        (a,) = np.flatnonzero(blk.loading) if np.any(blk.loading) else (None,)
        if a is not None:
            seen.append((a, blk.lam * blk.loading[a] ** 2, blk.sigma_t,
                         blk.spatial))
    per_comp = [sorted((lam, s, repr(k)) for c, lam, s, k in seen if c == a)
                for a in range(N)]
    return all(p == per_comp[0] for p in per_comp)


def admissible_flags(model: ModelSpec) -> dict:
    """Every expansion flag the model admits, switched on: R is always
    diagonal (``DiagonalA``), scalar when the rates are equal; C is diagonal
    or a multiple of the identity as :func:`_diagonal` and
    :func:`_isotropic` decide."""
    return dict(diag_R=True, iso_R=len(set(model.gamma)) == 1,
                diag_C=_diagonal(model), iso_C=_isotropic(model))


# ---------------------------------------------------------------------------
# Noise blocks
# ---------------------------------------------------------------------------

def _kernel(rng: np.random.Generator, kind: str) -> Kernel:
    u = rng.uniform
    params = {"constant": (),
              "exponential": (u(0.5, 1.5),),
              "gaussian": (u(0.5, 1.5),),
              "damped_cosine": (u(0.5, 1.5), u(0.5, 2.0)),
              "legendre": tuple(u(0.1, 0.6, size=3)),
              "inhomogeneous": (u(0.5, 1.5), u(-0.5, 0.5), u(0.8, 2.0))}
    return Kernel(kind, params[kind])


KERNELS = ("constant", "exponential", "gaussian", "damped_cosine", "legendre",
           "inhomogeneous")


def _sym(rng: np.random.Generator, N: int, m: int, scale: float) -> np.ndarray:
    return rng.normal(scale=scale, size=(N,) * m)


def _ou_general(rng, N):
    out = []
    for _ in range(rng.choice([0, 1, 1, 2])):
        loading = rng.normal(scale=0.7, size=N)
        if rng.random() < 0.3:
            keep = rng.integers(N)
            loading = np.where(np.arange(N) == keep, loading, 0.0)
        out.append(OUBlock(rng.uniform(0.2, 0.8), rng.uniform(0.3, 1.5),
                           _kernel(rng, str(rng.choice(KERNELS))),
                           tuple(loading)))
    return tuple(out)


def _ou_diagonal(rng, N):
    out = []
    for a in range(N):
        if rng.random() < 0.8:
            loading = np.zeros(N)
            loading[a] = rng.uniform(0.5, 1.2) * rng.choice([-1, 1])
            out.append(OUBlock(rng.uniform(0.2, 0.8), rng.uniform(0.3, 1.5),
                               _kernel(rng, str(rng.choice(KERNELS))),
                               tuple(loading)))
    return tuple(out)


def _ou_isotropic(rng, N):
    out = []
    for _ in range(rng.choice([0, 1, 1, 2])):
        lam, sigma = rng.uniform(0.2, 0.8), rng.uniform(0.3, 1.5)
        kernel = _kernel(rng, str(rng.choice(KERNELS)))
        out += [OUBlock(lam, sigma, kernel, tuple(np.eye(N)[a]))
                for a in range(N)]
    return tuple(out)


def _white(rng, N, kind):
    if rng.random() < 0.5:
        return None
    if kind == "general":
        A = rng.normal(scale=0.4, size=(N, N)).tolist()
        return tuple(tuple(math.fsum(a * b for a, b in zip(A[i], A[j]))
                           for j in range(N)) for i in range(N))
    if kind == "diagonal":
        return tuple(map(tuple, np.diag(rng.uniform(0.05, 0.4, size=N))))
    return tuple(map(tuple, rng.uniform(0.05, 0.4) * np.eye(N)))


def _multiplicative(rng, N, kind):
    if rng.random() > 0.25:
        return None
    interpretation = str(rng.choice(["ito", "stratonovich"]))
    if kind == "general":
        M = int(rng.integers(1, 3))
        g0 = rng.normal(scale=0.4, size=(N, M))
    elif kind == "diagonal":
        M = N
        g0 = np.diag(rng.uniform(0.2, 0.5, size=N))
    else:
        M = N
        g0 = rng.uniform(0.2, 0.5) * np.eye(N)
    g1 = rng.normal(scale=0.25, size=(N, M, N))
    return Multiplicative(g0, g1, interpretation)


def _poisson(rng, N, white_pulses_allowed):
    pulse = str(rng.choice(["white", "exponential"]))
    if pulse == "white" and not white_pulses_allowed:
        pulse = "exponential"
    orders = [(), (3,), (3, 4)][rng.choice(3, p=[0.2, 0.5, 0.3])]
    if pulse == "exponential":
        orders = orders[:1]
    tau = (tuple(rng.uniform(0.3, 1.5, size=N)) if pulse == "exponential"
           else None)
    return Poisson(rng.uniform(0.5, 1.5), tuple(rng.normal(scale=0.6, size=N)),
                   tuple(rng.uniform(0.4, 1.0, size=N)), pulse, tau, orders)


def _cumulants(rng, N):
    out = []
    for m in (1, 2, 3):
        if rng.random() < 0.25:
            out.append(Cumulant(m, _sym(rng, N, m, 0.3), "static"))
    for m in (2, 3):
        if rng.random() < 0.2:
            out.append(Cumulant(m, _sym(rng, N, m, 0.3), "white"))
    return tuple(out)


def _t_min(rng) -> float:
    t = rng.uniform(0.05, 0.6)
    return float(t if rng.random() < 0.5 else -t)


def _model(rng: np.random.Generator, kind: str) -> ModelSpec:
    N = int(rng.choice([2, 3]))
    if kind == "isotropic" or rng.random() < 0.1:
        gamma = (float(rng.uniform(0.4, 1.8)),) * N
    else:
        gamma = tuple(rng.uniform(0.4, 1.8, size=N))
    source = tuple(rng.normal(scale=0.3, size=N)) if rng.random() < 0.5 \
        else None
    quadratic = _sym(rng, N, 3, 0.3) if rng.random() < 0.8 else None
    cubic = _sym(rng, N, 4, 0.15) if rng.random() < 0.4 else None
    ou = {"general": _ou_general, "diagonal": _ou_diagonal,
          "isotropic": _ou_isotropic}[kind](rng, N)
    white = _white(rng, N, kind)
    mult = _multiplicative(rng, N, kind)
    poisson = (_poisson(rng, N, mult is None)
               if kind == "general" and rng.random() < 0.35 else None)
    cumulants = _cumulants(rng, N)
    if not (ou or white or mult or poisson or cumulants):
        white = tuple(map(tuple, 0.2 * np.eye(N)))
    return ModelSpec(gamma=gamma, source=source, quadratic=quadratic,
                     cubic=cubic, ou=ou, white=white, multiplicative=mult,
                     poisson=poisson, cumulants=cumulants, t_min=_t_min(rng))


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------

def _positions(rng, labels, d, shared):
    out = {}
    first = None
    for lab in labels:
        if shared and first is not None:
            out[lab] = first
            continue
        while True:
            v = rng.uniform(-1.0, 1.0, size=d)
            if np.linalg.norm(v) > 0.2:
                break
        out[lab] = tuple(float(c) for c in v) if d > 1 else float(v[0])
        first = out[lab]
    return out


def _observable(rng: np.random.Generator, model: ModelSpec,
                kind: str) -> Observable:
    N = model.n_components
    n_legs = int(rng.choice([1, 2, 3], p=[0.15, 0.6, 0.25]))
    labels = [str(lab) for lab in rng.permutation(list(LABELS[:n_legs]))]
    comps = rng.integers(N, size=n_legs)
    legendre = any(blk.spatial.kind == "legendre" for blk in model.ou)
    d = int(rng.choice([2, 3])) if legendre else int(rng.choice([1, 1, 2, 3]))
    positions = _positions(rng, labels, d, model.multiplicative is not None)
    span = rng.uniform(0.8, 1.8)
    t_final = model.t_min + span
    times, over = {}, ()
    if n_legs >= 2 and rng.random() < (0.6 if kind == "general" else 0.25):
        pinned = labels[rng.integers(n_legs)]
        times = {pinned: model.t_min + rng.uniform(0.3, 0.9) * span}
    if rng.random() < 0.15:
        free = [lab for lab in labels if lab not in times]
        over = (free[rng.integers(len(free))],)
    target = ("connected" if n_legs >= 2 and rng.random() < 0.15
              else "moment")
    return Observable([Leg(int(c), lab) for c, lab in zip(comps, labels)],
                      positions, t_final, times=times, integrate_over=over,
                      target=target)


def _max_weight(rng, model: ModelSpec, obs: Observable) -> int:
    u = rng.random()
    if u < 0.1:
        return 1
    if (u > 0.8 and len(obs.legs) <= 2 and model.cubic is None
            and len(model.channels()) <= 3):
        return 3
    return 2


def draw(seed: int, kind: str | None = None) -> Case:
    """The case of ``seed``; ``kind`` forces a kind of model."""
    rng = np.random.default_rng(seed)
    kinds = list(KINDS)
    chosen = str(rng.choice(kinds, p=[KINDS[k] for k in kinds]))
    kind = kind or chosen
    model = _model(rng, kind)
    obs = _observable(rng, model, kind)
    return Case(int(seed), kind, model, obs, _max_weight(rng, model, obs),
                admissible_flags(model))
