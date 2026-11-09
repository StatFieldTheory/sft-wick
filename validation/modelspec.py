r"""A neutral description of a stochastic field model and of an observable.

The description is data: frozen dataclasses of numbers and tuples, hashable
and convertible to JSON.  It refers to no sft-wick object, so the same
description can be compiled into the exact moment hierarchy
(:mod:`validation.hierarchy`) and lowered onto any sft-wick release.

The model is a field ``φ_a(x, t)``, ``a = 0 … N−1``, that starts at
``φ(t_min) = 0`` and obeys, at every position ``x``,

.. math::

    dφ_a = \bigl(−γ_a φ_a + h_a + F_{abc} φ_b φ_c + G_{abcd} φ_b φ_c φ_d
           + Σ_k L_{ka} ζ_k + η_a + Z_a\bigr)\,dt
           + dW_a + Σ_k g_{ak}(φ) \,(∘)\, dB_k + dJ_a .

Nothing couples two positions except the noise, so the response is local in
space.  The noises are independent of one another:

* ``ζ_k`` (:class:`OUBlock`): stationary Gaussian, ``⟨ζ_k(x,t) ζ_l(y,s)⟩ =
  δ_kl λ_k e^{−|t−s|/σ_k} K_k(x, y)``, loaded onto the components by
  ``L_k``;
* ``W`` (``white``): Gaussian white noise, ``⟨dW_a dW_b⟩ = S_ab dt``, the
  same process at every position;
* ``B_k`` (:class:`Multiplicative`): Wiener processes shared by every
  position, ``g_ak(φ) = g0_ak + g1_akb φ_b``, read as Itô or Stratonovich;
* ``η`` (:class:`Poisson`): compound-Poisson events at rate ``ν`` per unit
  volume and time, each driving component ``a`` at ``x`` by ``A_a
  e^{−|x−z|²/(2 s_a²)}`` times a time profile, ``δ`` (white pulses) or
  ``Θ(τ) e^{−τ/τ_a}`` (exponential pulses, from the infinite past), minus
  its mean.  Only the cumulants of order 2 and of the orders in
  ``Poisson.orders`` are kept, so this is the shot noise with its other
  cumulants set to zero;
* ``Z`` (:class:`Cumulant`, ``time='static'``): a random vector, constant
  in space and time, with joint cumulants ``κ^(m)``;
* ``J`` (:class:`Cumulant`, ``time='white'``): a compensated jump process
  (zero mean), the same at every position, whose rate-weighted jump
  moments are ``κ^(m)`` at the orders given and 0 at every other order.

A cumulant tensor enters symmetrised over its legs.  Every source of
non-linearity or non-Gaussianity is a *channel* (:meth:`ModelSpec.channels`):
``h``, ``F``, ``G``, ``g`` (one power per ``g1`` factor), ``S{m}`` and
``W{m}`` (one per static or white cumulant of order ``m``), and ``P{m}``
(one per Poisson cumulant of order ``m ≥ 3`` kept in the model).  A moment
is a polynomial in one bookkeeping parameter per channel.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.polynomial import legendre

__all__ = ["Kernel", "OUBlock", "Multiplicative", "Poisson", "Cumulant",
           "ModelSpec", "Leg", "Observable", "model_to_dict",
           "model_from_dict", "observable_to_dict", "observable_from_dict",
           "as_array"]

#: Number of parameters of each spatial kernel (``legendre``: one or more).
KERNEL_PARAMS = {"constant": 0, "exponential": 1, "gaussian": 1,
                 "damped_cosine": 2, "inhomogeneous": 3, "legendre": None}


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _nested(a: np.ndarray) -> Any:
    if a.ndim == 0:
        return float(a)
    return tuple(_nested(x) for x in a)


def _tensor(x: Any, shape: tuple, name: str) -> tuple:
    """``x`` as nested tuples of floats, checked against ``shape``."""
    a = np.asarray(x, dtype=float)
    if a.shape != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, "
                         f"got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    return _nested(a)


def as_array(x: Any) -> np.ndarray:
    """A stored tensor as a float array."""
    return np.asarray(x, dtype=float)


def _position(x: Any) -> tuple:
    a = np.atleast_1d(np.asarray(x, dtype=float))
    if a.ndim != 1 or not np.all(np.isfinite(a)):
        raise ValueError(f"a position is a finite scalar or vector, got {x!r}")
    return tuple(float(v) for v in a)


def _positive(value: float, name: str) -> float:
    v = float(value)
    if not (math.isfinite(v) and v > 0.0):
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return v


def _set(obj: Any, name: str, value: Any) -> None:
    object.__setattr__(obj, name, value)


# ---------------------------------------------------------------------------
# Noise blocks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Kernel:
    """A spatial kernel ``K(x, x')``; ``r = |x − x'|``.

    ``constant``: 1.  ``exponential``, ``(σ,)``: ``e^{−r/σ}``.
    ``gaussian``, ``(σ,)``: ``e^{−r²/2σ²}``.  ``damped_cosine``, ``(ℓ, k)``:
    ``e^{−r/ℓ} cos(k r)``.  ``legendre``, ``(C_0, C_1, …)``: ``Σ_ℓ C_ℓ
    P_ℓ(cos θ)`` with ``θ`` the angle between ``x`` and ``x'``.
    ``inhomogeneous``, ``(s, c, L)``: ``g(x) g(x') e^{−r²/2s²}`` with ``g(x) =
    e^{−|x − c|²/2L²}`` (``c`` on every coordinate).
    """

    kind: str
    params: tuple = ()

    def __post_init__(self) -> None:
        if self.kind not in KERNEL_PARAMS:
            raise ValueError(f"unknown kernel {self.kind!r}; expected one of "
                             f"{sorted(KERNEL_PARAMS)}")
        params = tuple(float(p) for p in np.atleast_1d(
            np.asarray(self.params, dtype=float)))
        expected = KERNEL_PARAMS[self.kind]
        if expected is None and not params:
            raise ValueError("a legendre kernel needs at least one coefficient")
        if expected is not None and len(params) != expected:
            raise ValueError(f"kernel {self.kind!r} takes {expected} "
                             f"parameters, got {len(params)}")
        if not all(math.isfinite(p) for p in params):
            raise ValueError("kernel parameters must be finite")
        if self.kind in ("exponential", "gaussian", "damped_cosine"):
            _positive(params[0], f"{self.kind} length")
        if self.kind == "inhomogeneous":
            _positive(params[0], "inhomogeneous width s")
            _positive(params[2], "inhomogeneous envelope length L")
        _set(self, "params", params)

    def __call__(self, x1: Any, x2: Any) -> float:
        a, b = np.asarray(_position(x1)), np.asarray(_position(x2))
        if a.shape != b.shape:
            raise ValueError("kernel positions differ in dimension")
        p = self.params
        r = float(np.sqrt(np.sum((a - b) ** 2)))
        if self.kind == "constant":
            return 1.0
        if self.kind == "exponential":
            return math.exp(-r / p[0])
        if self.kind == "gaussian":
            return math.exp(-r * r / (2.0 * p[0] ** 2))
        if self.kind == "damped_cosine":
            return math.exp(-r / p[0]) * math.cos(p[1] * r)
        if self.kind == "legendre":
            na, nb = float(np.sqrt(a @ a)), float(np.sqrt(b @ b))
            if na == 0.0 or nb == 0.0:
                raise ValueError("a legendre kernel needs non-zero vectors")
            cos = min(1.0, max(-1.0, float(a @ b) / (na * nb)))
            return float(legendre.legval(cos, p))
        s, c, length = p
        g = math.exp(-(float(np.sum((a - c) ** 2))
                       + float(np.sum((b - c) ** 2))) / (2.0 * length ** 2))
        return g * math.exp(-r * r / (2.0 * s ** 2))


@dataclass(frozen=True)
class OUBlock:
    """Stationary coloured noise ``ζ`` with covariance ``lam
    e^{−|t−s|/sigma_t} K(x, y)``, ``K = spatial``, so its variance at ``x``
    is ``lam K(x, x)``; ``dφ_a ⊃ loading[a] ζ dt``."""

    lam: float
    sigma_t: float
    spatial: Kernel
    loading: tuple

    def __post_init__(self) -> None:
        lam = float(self.lam)
        if not (math.isfinite(lam) and lam >= 0.0):
            raise ValueError(f"lam must be non-negative, got {self.lam!r}")
        _set(self, "lam", lam)
        _set(self, "sigma_t", _positive(self.sigma_t, "sigma_t"))
        if not isinstance(self.spatial, Kernel):
            raise TypeError("spatial must be a Kernel")
        loading = np.atleast_1d(np.asarray(self.loading, dtype=float))
        _set(self, "loading", _tensor(loading, loading.shape, "loading"))


@dataclass(frozen=True)
class Multiplicative:
    """White noise ``g_ak(φ) dB_k`` with ``g_ak = g0[a][k] + Σ_b g1[a][k][b]
    φ_b``; the ``B_k`` are shared by every position."""

    g0: tuple
    g1: tuple
    interpretation: str = "ito"

    def __post_init__(self) -> None:
        g0 = np.asarray(self.g0, dtype=float)
        if g0.ndim != 2:
            raise ValueError("g0 must have shape (N, M)")
        N, M = g0.shape
        _set(self, "g0", _tensor(g0, (N, M), "g0"))
        _set(self, "g1", _tensor(self.g1, (N, M, N), "g1"))
        if self.interpretation not in ("ito", "stratonovich"):
            raise ValueError("interpretation must be 'ito' or "
                             f"'stratonovich', got {self.interpretation!r}")

    @property
    def has_g1(self) -> bool:
        return bool(np.any(as_array(self.g1)))


@dataclass(frozen=True)
class Poisson:
    """Compound-Poisson shot noise minus its mean (module docstring).

    ``orders`` lists the cumulant orders ``m ≥ 3`` kept in the model; the
    Gaussian part (``m = 2``) is always kept, and every other order is 0.
    Each kept order is a channel ``P{m}``.
    """

    nu: float
    amplitude: tuple
    width: tuple
    pulse: str = "white"
    tau: tuple | None = None
    orders: tuple = (3,)

    def __post_init__(self) -> None:
        _set(self, "nu", _positive(self.nu, "nu"))
        amp = np.atleast_1d(np.asarray(self.amplitude, dtype=float))
        N = amp.shape[0]
        _set(self, "amplitude", _tensor(amp, (N,), "amplitude"))
        width = _tensor(self.width, (N,), "width")
        for w in width:
            _positive(w, "width")
        _set(self, "width", width)
        if self.pulse not in ("white", "exponential"):
            raise ValueError(f"pulse must be 'white' or 'exponential', "
                             f"got {self.pulse!r}")
        if self.pulse == "exponential":
            if self.tau is None:
                raise ValueError("exponential pulses need tau")
            tau = _tensor(self.tau, (N,), "tau")
            for t in tau:
                _positive(t, "tau")
            _set(self, "tau", tau)
        elif self.tau is not None:
            raise ValueError("white pulses take no tau")
        orders = tuple(sorted({int(m) for m in self.orders}))
        if any(m < 3 for m in orders):
            raise ValueError("Poisson orders are cumulant orders m >= 3")
        _set(self, "orders", orders)


@dataclass(frozen=True)
class Cumulant:
    """A static (``time='static'``, order ``m ≥ 1``) or white
    (``time='white'``, ``m ≥ 2``) cumulant ``κ^(m)``, shape ``(N,)*m``."""

    order: int
    tensor: tuple
    time: str = "static"

    def __post_init__(self) -> None:
        if self.time not in ("static", "white"):
            raise ValueError(f"time must be 'static' or 'white', "
                             f"got {self.time!r}")
        m = int(self.order)
        if m < (1 if self.time == "static" else 2):
            raise ValueError(f"a {self.time} cumulant has order >= "
                             f"{1 if self.time == 'static' else 2}")
        _set(self, "order", m)
        t = np.asarray(self.tensor, dtype=float)
        if t.ndim != m or len(set(t.shape)) > 1:
            raise ValueError(f"an order-{m} cumulant has shape (N,)*{m}")
        _set(self, "tensor", _tensor(t, t.shape, "cumulant tensor"))

    @property
    def channel(self) -> str:
        return f"{'S' if self.time == 'static' else 'W'}{self.order}"


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """A model (module docstring).  Tensors may be given as arrays; they
    are stored as nested tuples."""

    gamma: tuple
    source: tuple | None = None
    quadratic: tuple | None = None
    cubic: tuple | None = None
    ou: tuple = ()
    white: tuple | None = None
    multiplicative: Multiplicative | None = None
    poisson: Poisson | None = None
    cumulants: tuple = ()
    t_min: float = 0.0

    def __post_init__(self) -> None:
        gamma = np.atleast_1d(np.asarray(self.gamma, dtype=float))
        N = gamma.shape[0]
        if gamma.ndim != 1 or N < 1:
            raise ValueError("gamma must be a non-empty vector")
        _set(self, "gamma", _tensor(gamma, (N,), "gamma"))
        for name, rank in (("source", 1), ("quadratic", 3), ("cubic", 4)):
            value = getattr(self, name)
            if value is not None:
                _set(self, name, _tensor(value, (N,) * rank, name))
        _set(self, "ou", tuple(self.ou))
        for block in self.ou:
            if not isinstance(block, OUBlock):
                raise TypeError("ou holds OUBlock objects")
            if len(block.loading) != N:
                raise ValueError(f"an OU loading has {N} entries")
        if self.white is not None:
            _set(self, "white", _symmetric_psd(self.white, N, "white"))
        if self.multiplicative is not None:
            if len(self.multiplicative.g0) != N:
                raise ValueError(f"multiplicative g0 has {N} rows")
        if self.poisson is not None and len(self.poisson.amplitude) != N:
            raise ValueError(f"Poisson amplitude has {N} entries")
        _set(self, "cumulants", tuple(self.cumulants))
        seen = set()
        for c in self.cumulants:
            if not isinstance(c, Cumulant):
                raise TypeError("cumulants holds Cumulant objects")
            if len(c.tensor) != N:
                raise ValueError(f"cumulant {c.channel} must have N = {N}")
            if c.channel in seen:
                raise ValueError(f"two cumulants of channel {c.channel}")
            seen.add(c.channel)
        t_min = float(self.t_min)
        if not math.isfinite(t_min):
            raise ValueError("t_min must be finite")
        _set(self, "t_min", t_min)

    @property
    def n_components(self) -> int:
        return len(self.gamma)

    def channels(self) -> tuple:
        """The channel names, in the order of the hierarchy's tags."""
        out = []
        if self.source is not None:
            out.append("h")
        if self.quadratic is not None:
            out.append("F")
        if self.cubic is not None:
            out.append("G")
        if self.multiplicative is not None and self.multiplicative.has_g1:
            out.append("g")
        for time in ("static", "white"):
            out += sorted((c.channel for c in self.cumulants if c.time == time),
                          key=lambda name: int(name[1:]))
        if self.poisson is not None:
            out += [f"P{m}" for m in self.poisson.orders]
        return tuple(out)


def _symmetric_psd(x: Any, N: int, name: str) -> tuple:
    S = as_array(_tensor(x, (N, N), name))
    scale = max(1.0, float(np.max(np.abs(S))))
    if np.max(np.abs(S - S.T)) > 1e-12 * scale:
        raise ValueError(f"{name} must be symmetric")
    S = 0.5 * (S + S.T)
    if np.min(np.linalg.eigvalsh(S)) < -1e-12 * scale:
        raise ValueError(f"{name} must be positive semi-definite")
    return _nested(S)


# ---------------------------------------------------------------------------
# The observable
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Leg:
    """One external operator: ``op`` (``'phi'`` or ``'psi'``) of component
    ``comp`` at the point named ``label``."""

    comp: int
    label: str
    op: str = "phi"

    def __post_init__(self) -> None:
        if self.op not in ("phi", "psi"):
            raise ValueError(f"op must be 'phi' or 'psi', got {self.op!r}")
        if int(self.comp) < 0:
            raise ValueError("comp must be non-negative")
        _set(self, "comp", int(self.comp))
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("label must be a non-empty string")


@dataclass(frozen=True)
class Observable:
    """``⟨Π_legs op_comp(label)⟩`` (``target='moment'``) or its joint
    cumulant (``target='connected'``).

    Every leg sits at ``positions[label]``.  A leg's time is
    ``times[label]`` when given, else ``t_final``; a leg in
    ``integrate_over`` (or every leg, for ``'all'``) is the time integral of
    the field over ``[t_min, t_final]``.  Labels are distinct, as sft-wick
    requires; two labels may share a position.
    """

    legs: tuple
    positions: Any
    t_final: float
    times: Any = ()
    integrate_over: Any = ()
    target: str = "moment"

    def __post_init__(self) -> None:
        legs = tuple(leg if isinstance(leg, Leg) else Leg(*leg)
                     for leg in self.legs)
        if not legs:
            raise ValueError("an observable has at least one leg")
        labels = [leg.label for leg in legs]
        if len(set(labels)) != len(labels):
            raise ValueError("leg labels must be distinct")
        _set(self, "legs", legs)
        positions = dict(self.positions)
        missing = [lab for lab in labels if lab not in positions]
        if missing:
            raise ValueError(f"no position for {missing}")
        pos = tuple((lab, _position(positions[lab])) for lab in sorted(labels))
        if len({len(p) for _, p in pos}) != 1:
            raise ValueError("positions differ in dimension")
        _set(self, "positions", pos)
        t_final = float(self.t_final)
        if not math.isfinite(t_final):
            raise ValueError("t_final must be finite")
        _set(self, "t_final", t_final)
        over = self.integrate_over
        over = tuple(sorted(labels)) if over == "all" else tuple(sorted(over))
        if not set(over) <= set(labels):
            raise ValueError(f"integrate_over names unknown labels {over}")
        _set(self, "integrate_over", over)
        times = dict(self.times)
        if not set(times) <= set(labels):
            raise ValueError("times names unknown labels")
        if set(times) & set(over):
            raise ValueError("an integrated leg takes no pinned time")
        for lab, t in times.items():
            if not float(t) <= t_final:
                raise ValueError(f"time of {lab!r} is after t_final")
        _set(self, "times", tuple((lab, float(times[lab]))
                                  for lab in sorted(times)))
        if self.target not in ("moment", "connected"):
            raise ValueError(f"target must be 'moment' or 'connected', "
                             f"got {self.target!r}")

    def position(self, label: str) -> tuple:
        return dict(self.positions)[label]

    def time(self, label: str) -> float:
        return dict(self.times).get(label, self.t_final)


# ---------------------------------------------------------------------------
# JSON round trip
# ---------------------------------------------------------------------------

def _plain(x: Any) -> Any:
    if isinstance(x, tuple):
        return [_plain(v) for v in x]
    return x


def model_to_dict(spec: ModelSpec) -> dict:
    """``spec`` as JSON-ready dicts and lists."""
    out: dict = {"gamma": _plain(spec.gamma), "t_min": spec.t_min}
    for name in ("source", "quadratic", "cubic", "white"):
        value = getattr(spec, name)
        if value is not None:
            out[name] = _plain(value)
    if spec.ou:
        out["ou"] = [dict(lam=b.lam, sigma_t=b.sigma_t,
                          spatial=dict(kind=b.spatial.kind,
                                       params=list(b.spatial.params)),
                          loading=list(b.loading)) for b in spec.ou]
    m = spec.multiplicative
    if m is not None:
        out["multiplicative"] = dict(g0=_plain(m.g0), g1=_plain(m.g1),
                                     interpretation=m.interpretation)
    p = spec.poisson
    if p is not None:
        out["poisson"] = dict(nu=p.nu, amplitude=list(p.amplitude),
                              width=list(p.width), pulse=p.pulse,
                              tau=None if p.tau is None else list(p.tau),
                              orders=list(p.orders))
    if spec.cumulants:
        out["cumulants"] = [dict(order=c.order, tensor=_plain(c.tensor),
                                 time=c.time) for c in spec.cumulants]
    return out


def model_from_dict(d: Mapping) -> ModelSpec:
    """The inverse of :func:`model_to_dict`."""
    kw = {k: d[k] for k in ("gamma", "source", "quadratic", "cubic", "white",
                            "t_min") if k in d}
    kw["ou"] = tuple(OUBlock(lam=b["lam"], sigma_t=b["sigma_t"],
                             spatial=Kernel(**b["spatial"]),
                             loading=b["loading"]) for b in d.get("ou", ()))
    if "multiplicative" in d:
        kw["multiplicative"] = Multiplicative(**d["multiplicative"])
    if "poisson" in d:
        kw["poisson"] = Poisson(**d["poisson"])
    kw["cumulants"] = tuple(Cumulant(**c) for c in d.get("cumulants", ()))
    return ModelSpec(**kw)


def observable_to_dict(obs: Observable) -> dict:
    return dict(legs=[[leg.comp, leg.label, leg.op] for leg in obs.legs],
                positions={lab: list(p) for lab, p in obs.positions},
                t_final=obs.t_final, times=dict(obs.times),
                integrate_over=list(obs.integrate_over), target=obs.target)


def observable_from_dict(d: Mapping) -> Observable:
    return Observable(legs=tuple(Leg(*leg) for leg in d["legs"]),
                      positions=d["positions"], t_final=d["t_final"],
                      times=d.get("times", ()),
                      integrate_over=tuple(d.get("integrate_over", ())),
                      target=d.get("target", "moment"))
