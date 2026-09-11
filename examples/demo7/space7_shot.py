r"""Demo 7, item (d): compound-Poisson shot noise in three dimensions.

Events ``(z_k, s_k)``, ``z_k ∈ R^3``, form a Poisson process of rate ``ν``
per unit volume per unit time.  Each event drives component ``a`` at
``x ∈ R^3`` by

.. math::

    η_a(x, t) = Σ_k h_a\, w_a(x − z_k)\, g_a(t − s_k) − ⟨·⟩,
    \qquad w_a(u) = e^{−|u|^2/(2 s_a^2)},

with ``g_a = δ`` (white pulses) or ``g_a(τ) = Θ(τ) e^{−τ/τ_a}``
(exponential pulses, from the infinite past).  Campbell's theorem gives
every cumulant as one source-point integral,

.. math::

    κ^{(m)}_{a_1…a_m}(x_1, t_1; …) = ν \prod_j h_{a_j}\, X_a(x)\, G_a(t),
    \qquad X_a(x) = ∫ d^3z \prod_j w_{a_j}(x_j − z)
                  = \Bigl(\frac{2π}{P}\Bigr)^{3/2} e^{−\frac12 Σ_j w_j |x_j − \bar x|^2},

``w_j = 1/s_{a_j}^2``, ``P = Σ_j w_j``, ``\bar x = Σ_j w_j x_j / P``: a
Gaussian overlap integral in closed form.  The field obeys ``dφ_a/dt =
−γ_a φ_a + F_abc φ_b φ_c + η_a`` from ``φ = 0`` at ``t_min``; contracting
``κ^(m)``'s legs with ``R`` keeps the single source time, so the
R-contracted cumulant is ``K_R = ν Π h · X · T̃`` (demo 4 derives ``T̃``;
here the times are measured from ``t_min`` and white pulses allow a rate
per component).  ``K_R`` at ``m = 2`` is the exact C propagator.

The second half of the module is the exact reference: the moment hierarchy
of the Markov process at the observation points (demo 4's, with 3-D
positions and per-component rates).  numpy and
``examples/reference/ito_moments.py`` only; no sft-wick code.
"""
from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import ito_moments as im  # noqa: E402

__all__ = ["ShotParams", "PARAMS_WHITE", "PARAMS_EXP", "PARAMS_WHITE_MATRIX",
           "F_SHOT", "POINTS", "spatial_overlap", "t_tilde", "K_R",
           "kappa_raw", "C_matrix", "ShotHierarchy"]


@dataclass(frozen=True)
class ShotParams:
    """Parameters of the 3-D shot noise.

    Args:
        nu: events per unit volume per unit time.
        h: amplitude per component (of both signs).
        s: 3-D Gaussian width per component.
        tau: pulse decay time per component (exponential pulses).
        gamma: decay rate per component; exponential pulses need them equal.
        pulse: ``"white"`` or ``"exponential"``.
        t_min: start of the field's evolution.
    """

    nu: float = 1.2
    h: tuple = (0.9, -0.6)
    s: tuple = (0.5, 0.9)
    tau: tuple = (0.35, 1.8)
    gamma: tuple = (1.0, 1.0)
    pulse: str = "white"
    t_min: float = 0.3

    def __post_init__(self) -> None:
        if self.pulse not in ("white", "exponential"):
            raise ValueError(f"pulse must be 'white' or 'exponential', "
                             f"got {self.pulse!r}")
        if self.pulse == "exponential":
            if len(set(self.gamma)) != 1:
                raise ValueError("exponential pulses: the closed form is "
                                 "written for one decay rate")
            g = self.gamma[0]
            for r in self.rates:
                if abs(r - g) < 0.2 * max(r, g):
                    raise ValueError("pulse rate 1/tau_a too close to gamma")

    @property
    def n_components(self) -> int:
        return len(self.h)

    @property
    def rates(self) -> tuple:
        return tuple(1.0 / t for t in self.tau)


PARAMS_WHITE = ShotParams(pulse="white")
PARAMS_EXP = ShotParams(pulse="exponential")
PARAMS_WHITE_MATRIX = ShotParams(pulse="white", gamma=(0.8, 1.3))

#: ``dφ_a/dt ⊃ F_abc φ_b φ_c``; no index symmetry.
F_SHOT = np.array([[[0.20, -0.35], [0.25, 0.10]],
                   [[-0.30, 0.15], [0.40, -0.25]]])

#: Three points in R^3, no two on a coordinate axis together.
POINTS = {"x": (0.0, 0.0, 0.0), "y": (0.5, -0.3, 0.4),
          "z": (-0.2, 0.6, -0.35)}


# ---------------------------------------------------------------------------
# Closed forms
# ---------------------------------------------------------------------------

def _legs_x(x) -> np.ndarray:
    """``(m, d)`` or ``(m, n, d)`` leg positions as ``(m, n, d)``."""
    x = np.asarray(x, dtype=float)
    return x[:, None, :] if x.ndim == 2 else x


def _legs_t(t) -> np.ndarray:
    """``(m,)`` or ``(m, n)`` leg times as ``(m, n)``."""
    t = np.asarray(t, dtype=float)
    return t[:, None] if t.ndim == 1 else t


def spatial_overlap(comps, xs, p: ShotParams, powers=None) -> np.ndarray:
    r"""``X = ∫d^dz Π_j w_{a_j}(x_j − z)^{k_j}``, shape ``(n,)``.

    ``xs`` is ``(m, d)`` or ``(m, n, d)``; ``powers`` (default 1) raises each
    factor to ``k_j``, which the hierarchy's jump moments need."""
    xs = _legs_x(xs)
    d = xs.shape[-1]
    k = np.ones(len(comps)) if powers is None else np.asarray(powers, float)
    wt = np.array([kj / p.s[a] ** 2 for a, kj in zip(comps, k)])
    P = wt.sum()
    mean = np.einsum("j,jnd->nd", wt, xs) / P
    spread = np.einsum("j,jn->n", wt, ((xs - mean[None]) ** 2).sum(axis=-1))
    return (2.0 * np.pi / P) ** (d / 2.0) * np.exp(-0.5 * spread)


def t_tilde(comps, ts, p: ShotParams) -> np.ndarray:
    r"""``T̃_a(t') = ∫ds Π_j J_{a_j}(t'_j, s)`` in closed form, ``(n,)``;
    ``J_a(t, s) = ∫_{t_min}^t R_a(t, u) g_a(u − s) du``.

    Times ``τ_j = t_j − t_min``, ``T = min_j τ_j``.  White pulses, per-leg
    rates ``γ_j``: ``e^{−Σ_j γ_j (τ_j − T)} (1 − e^{−Γ T})/Γ``, ``Γ = Σ_j γ_j``.
    Exponential pulses (one ``γ``): demo 4's formula, the ``s < t_min``
    part ``Π_j A_j(τ_j)/Σ_j r_j`` plus the subset expansion of the
    ``t_min ≤ s ≤ T`` part.
    """
    tau = _legs_t(ts) - p.t_min
    m = tau.shape[0]
    T = tau.min(axis=0)
    positive = T > 0.0
    Tp = np.where(positive, T, 0.0)
    if p.pulse == "white":
        g = np.array([p.gamma[a] for a in comps])[:, None]
        Gam = float(g.sum())
        out = (np.exp(-(g * (tau - Tp)).sum(axis=0))
               * (-np.expm1(-Gam * Tp)) / Gam)
        return np.where(positive, out, 0.0)
    g = p.gamma[0]
    r = np.array([p.rates[a] for a in comps])
    A = (np.exp(-r[:, None] * tau) - np.exp(-g * tau)) / (g - r)[:, None]
    before = np.prod(A, axis=0) / r.sum()
    after = np.zeros_like(Tp)
    for mask in itertools.product((False, True), repeat=m):
        sel = np.array(mask)
        k = int(sel.sum())
        b = r[sel].sum() + g * (m - k)
        c = (r[sel][:, None] * tau[sel]).sum(axis=0) + g * tau[~sel].sum(axis=0)
        E = c - b * Tp
        after = after + (-1.0) ** (m - k) * (np.exp(-E) - np.exp(-c)) / b
    after = after / np.prod(g - r)
    return np.where(positive, before + after, 0.0)


def K_R(comps, xs, ts, p: ShotParams) -> np.ndarray:
    """The R-contracted cumulant ``ν Π h · X · T̃``, ``(n,)``."""
    amp = p.nu * np.prod([p.h[a] for a in comps])
    return amp * spatial_overlap(comps, xs, p) * t_tilde(comps, ts, p)


def kappa_raw(comps, xs, ts, p: ShotParams) -> np.ndarray:
    """The raw cumulant of white pulses: the amplitude ``ν Π h · X`` of
    ``δ(t_1 − t_2) … δ(t_{m−1} − t_m)`` (the vertex's ``equal_time=True``).
    ``ts`` is only used for its shape."""
    if p.pulse != "white":
        raise ValueError("the raw route is used for white pulses only")
    amp = p.nu * np.prod([p.h[a] for a in comps])
    out = amp * spatial_overlap(comps, xs, p)
    return np.broadcast_to(out, _legs_t(ts).shape[1:]).copy()


def C_matrix(x1, t1, x2, t2, p: ShotParams) -> np.ndarray:
    """The exact C, ``K_R`` at ``m = 2``: ``(n, N, N)``.  Positions ``(d,)``
    or ``(n, d)``, times scalar or ``(n,)``."""
    t1 = np.atleast_1d(np.asarray(t1, float))
    t2 = np.atleast_1d(np.asarray(t2, float))
    x1 = np.asarray(x1, float)
    x2 = np.asarray(x2, float)
    n = max(t1.shape[0], t2.shape[0],
            x1.shape[0] if x1.ndim == 2 else 1,
            x2.shape[0] if x2.ndim == 2 else 1)
    d = x1.shape[-1]
    xs = np.stack([np.broadcast_to(x1, (n, d)), np.broadcast_to(x2, (n, d))])
    ts = np.stack([np.broadcast_to(t1, (n,)), np.broadcast_to(t2, (n,))])
    N = p.n_components
    out = np.empty((n, N, N))
    for a in range(N):
        for b in range(N):
            out[:, a, b] = K_R((a, b), xs, ts, p)
    return out


# ---------------------------------------------------------------------------
# The exact reference: moment hierarchy at the observation points
# ---------------------------------------------------------------------------

class ShotHierarchy:
    r"""Moment hierarchy at the points ``positions`` (``(P, d)``).

    White pulses: the state is ``φ_{a,i}``, and an event at ``z`` moves it by
    ``h_a w_a(x_i − z)``.  Exponential pulses: ``(φ_{a,i}, η_{a,i})`` with
    ``dη = −η/τ_a dt`` plus the jumps, ``η`` started in its stationary law.
    Drift ``−γ_a φ_{a,i} (+ η_{a,i}) + ε F_abc φ_{b,i} φ_{c,i}``.  Tags
    ``(k, j)`` count ``ε^k`` and ``μ^j``, a jump cumulant of order ``m``
    carrying ``μ^{m−2}``: the tag ``(1, 1)`` of ``⟨φφ⟩`` is the package's
    FK3 channel.  Times are measured from ``t_min``.
    """

    def __init__(self, p: ShotParams, positions, f_tensor=F_SHOT,
                 max_jump_order: int = 5):
        self.p = p
        self.positions = np.asarray(positions, dtype=float)
        N, P = p.n_components, self.positions.shape[0]
        colored = p.pulse == "exponential"
        self.D = (2 if colored else 1) * N * P
        self._phi = {(a, i): a * P + i for a in range(N) for i in range(P)}
        self._eta = ({(a, i): N * P + a * P + i
                      for a in range(N) for i in range(P)} if colored else {})
        sde = im.PolySDE(D=self.D, n_tags=2)
        A = np.zeros((self.D, self.D))
        for (a, i), k in self._phi.items():
            A[k, k] = -p.gamma[a]
            if colored:
                A[k, self._eta[(a, i)]] = 1.0
        for (a, i), k in self._eta.items():
            A[k, k] = -p.rates[a]
        sde.add_linear_drift(A)
        Q = np.zeros((self.D,) * 3)
        for i in range(P):
            for a, b, c in itertools.product(range(N), repeat=3):
                Q[self._phi[(a, i)], self._phi[(b, i)],
                  self._phi[(c, i)]] += f_tensor[a, b, c]
        sde.add_quadratic_drift(Q, tag=(1, 0))
        jumped = self._eta if colored else self._phi
        self._jumped_index = {k: key for key, k in jumped.items()}
        sde.jump_moment = self._jump_moment
        sde.max_jump_order = max_jump_order
        sde.jump_tag = lambda m: (0, max(m - 2, 0))
        self.sde = sde

    def _jump_moment(self, gamma) -> float:
        comps, xs, powers = [], [], []
        amp = self.p.nu
        for k, g in enumerate(gamma):
            if g == 0:
                continue
            key = self._jumped_index.get(k)
            if key is None:          # a φ variable under exponential pulses
                return 0.0
            a, i = key
            comps.append(a)
            xs.append(self.positions[i])
            powers.append(g)
            amp *= self.p.h[a] ** g
        return float(amp * spatial_overlap(comps, np.array(xs), self.p,
                                           powers=powers)[0])

    def _initial(self, alpha, tag) -> float:
        zero = (0,) * self.D
        if tag[0] != 0:
            return 0.0
        if alpha == zero:
            return 1.0 if tag == (0, 0) else 0.0
        if not self._eta or any(alpha[k] for k in self._phi.values()):
            return 0.0
        legs = []
        for key, k in self._eta.items():
            legs += [key] * alpha[k]
        return _stationary_moment(tuple(legs), tag[1],
                                  tuple(map(tuple, self.positions)), self.p)

    def moments(self, legs, tags, T: float) -> dict:
        """``{tag: E[Π φ_{a,i}]^{(tag)}(T)}``, legs ``[(a, i), …]``, ``T``
        measured from ``t_min``."""
        mono = im.monomial_of(self.D, [self._phi[leg] for leg in legs])
        out = im.solve(self.sde, [(mono, tuple(t)) for t in tags], [T],
                       initial=self._initial)
        return {tuple(t): float(out[(mono, tuple(t))][0]) for t in tags}


def stationary_eta_cumulant(legs, positions, p: ShotParams) -> float:
    r"""Joint cumulant of ``η_{a_1}(x_{i_1}), …`` at one time, stationary
    exponential pulses: ``ν Π h · X / Σ_j r_{a_j}`` for ``m ≥ 2``."""
    if len(legs) < 2:
        return 0.0
    comps = [a for a, _ in legs]
    xs = np.array([positions[i] for _, i in legs], float)
    amp = p.nu * np.prod([p.h[a] for a in comps])
    return float(amp * spatial_overlap(comps, xs, p)[0]
                 / sum(p.rates[a] for a in comps))


@lru_cache(maxsize=None)
def _stationary_moment(legs, j, positions, p) -> float:
    """``E[Π η]`` at tag ``μ^j`` from the stationary cumulants: the sum over
    set partitions into blocks of size ≥ 2 with ``Σ (|b| − 2) = j``."""
    total = 0.0
    for partition in im._set_partitions(list(range(len(legs)))):
        if any(len(b) < 2 for b in partition):
            continue
        if sum(len(b) - 2 for b in partition) != j:
            continue
        term = 1.0
        for block in partition:
            term *= stationary_eta_cumulant([legs[k] for k in block],
                                            positions, p)
        total += term
    return total
