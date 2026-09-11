r"""Demo 4's driving noise: compound-Poisson events that drive every
component with its own amplitude, width and time profile.

Events ``(z_k, s_k)`` form a Poisson process of rate ``ν`` per unit length
per unit time.  Each event drives component ``a`` at point ``x`` by

.. math::

    η_a(x, t) = Σ_k h_a\, w_a(x − z_k)\, g_a(t − s_k) − ⟨·⟩,
    \qquad w_a(u) = e^{−u^2/(2 s_a^2)},

with one of two time profiles:

* ``pulse="white"``: ``g_a(τ) = δ(τ)``, instantaneous jumps;
* ``pulse="exponential"``: ``g_a(τ) = Θ(τ) e^{−τ/τ_a}``, from the infinite
  past, so the noise is stationary.

Campbell's theorem gives every cumulant as one source-point integral:

.. math::

    κ^{(m)}_{a_1…a_m}(x_1, t_1; …; x_m, t_m)
        = ν \prod_j h_{a_j}\; X_a(x)\; G_a(t),
    \qquad X_a(x) = ∫dz \prod_j w_{a_j}(x_j − z),
    \qquad G_a(t) = ∫ds \prod_j g_{a_j}(t_j − s).

The components share the events but not the pulse, so ``κ^(m)`` is
symmetric only when (component, point) pairs are permuted together.  At
fixed components it is not symmetric in the points (``s_a`` and ``τ_a``
differ), and at fixed points it is not symmetric in the components.

The field obeys ``dφ_a/dt = −γ φ_a + F_abc φ_b φ_c + η_a`` from ``φ = 0`` at
``t = 0``, so the response is ``R(t, u) = Θ(t − u) e^{−γ(t−u)}`` for every
component.  Contracting ``κ^(m)``'s legs with R keeps the single source
time, and the R-contracted cumulant is

.. math::

    K_R(a; x', t') = ν \prod_j h_{a_j}\, X_a(x')\, \tilde T_a(t'),
    \qquad \tilde T_a(t') = ∫ds \prod_j J_{a_j}(t'_j, s),
    \qquad J_a(t, s) = ∫_0^t R(t, u) g_a(u − s)\, du .

At ``F = 0`` this is the exact connected ``m``-point function of the field,
and ``K_R`` at ``m = 2`` is the exact C propagator.  This module uses numpy
only.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

__all__ = ["Params", "PARAMS_WHITE", "PARAMS_EXP", "F_TENSOR",
           "spatial_overlap", "raw_temporal", "t_tilde", "K_R", "kappa_raw",
           "C_matrix", "jump_moment", "stationary_eta_cumulant"]


@dataclass(frozen=True)
class Params:
    """Model parameters.

    Args:
        nu: event rate per unit length per unit time.
        h: amplitude of each component (one negative: the two components
            are skewed in opposite directions).
        s: spatial width of each component's pulse.
        tau: decay time of each component's pulse (``pulse="exponential"``).
        gamma: linear decay rate, common to the components (scalar R).
        pulse: ``"white"`` or ``"exponential"``.
    """

    nu: float = 1.5
    h: tuple = (1.0, -0.7)
    s: tuple = (0.6, 1.1)
    tau: tuple = (0.3, 2.0)
    gamma: float = 1.0
    pulse: str = "exponential"

    def __post_init__(self) -> None:
        if self.pulse not in ("white", "exponential"):
            raise ValueError(f"pulse must be 'white' or 'exponential', "
                             f"got {self.pulse!r}")
        if self.pulse == "exponential":
            for r in self.rates:
                if abs(r - self.gamma) < 0.2 * max(r, self.gamma):
                    raise ValueError(
                        "pulse rate 1/tau_a too close to gamma: the closed "
                        "form divides by (gamma - 1/tau_a)")

    @property
    def n_components(self) -> int:
        return len(self.h)

    @property
    def rates(self) -> tuple:
        return tuple(1.0 / t for t in self.tau)


PARAMS_WHITE = Params(pulse="white")
PARAMS_EXP = Params(pulse="exponential")

#: The drift tensor: ``dφ_a/dt ⊃ F_abc φ_b φ_c``.  No index symmetry (not
#: even in the last two indices, which the package must symmetrise itself).
F_TENSOR = np.array([[[0.25, -0.40], [0.15, 0.30]],
                     [[-0.35, 0.20], [0.45, -0.10]]])


def _legs(x) -> np.ndarray:
    """``(m,)`` or ``(m, n)`` leg data as ``(m, n)``."""
    x = np.asarray(x, dtype=float)
    return x[:, None] if x.ndim == 1 else x


def spatial_overlap(comps, xs, p: Params, powers=None) -> np.ndarray:
    r"""``X = ∫dz Π_j w_{a_j}(x_j − z)^{k_j}`` (Gaussian integral), ``(n,)``.

    ``xs`` is ``(m, n)``; ``powers`` (default all 1) raises each factor to
    ``k_j``, which is what the jump moments of the moment hierarchy need.
    """
    xs = _legs(xs)
    k = np.ones(len(comps)) if powers is None else np.asarray(powers, float)
    wt = np.array([kj / p.s[a] ** 2 for a, kj in zip(comps, k)])[:, None]
    P = wt.sum(axis=0)
    Q = (wt * xs).sum(axis=0)
    mean = Q / P
    spread = (wt * (xs - mean) ** 2).sum(axis=0)
    return np.sqrt(2.0 * np.pi / P) * np.exp(-0.5 * spread)


def raw_temporal(comps, ts, p: Params) -> np.ndarray:
    r"""``G_a(t) = ∫ds Π_j Θ(t_j − s) e^{−(t_j − s)/τ_{a_j}}``, exponential
    pulses only: ``e^{−Σ_j r_j (t_j − T)} / Σ_j r_j`` with ``T = min_j t_j``."""
    ts = _legs(ts)
    r = np.array([p.rates[a] for a in comps])[:, None]
    T = ts.min(axis=0)
    return np.exp(-(r * (ts - T)).sum(axis=0)) / r.sum()


def kappa_raw(comps, xs, ts, p: Params) -> np.ndarray:
    """The raw cumulant ``κ^(m)_{a}`` at the legs, ``(n,)``.  For white
    pulses this is the equal-time amplitude (the time deltas are the
    vertex's ``equal_time=True``)."""
    amp = p.nu * np.prod([p.h[a] for a in comps])
    val = amp * spatial_overlap(comps, xs, p)
    if p.pulse == "exponential":
        val = val * raw_temporal(comps, ts, p)
    return val


def t_tilde(comps, ts, p: Params) -> np.ndarray:
    r"""``T̃_a(t') = ∫ds Π_j J_{a_j}(t'_j, s)`` in closed form, ``(n,)``.

    White pulses: ``J(t, s) = e^{−γ(t−s)}`` for ``0 ≤ s ≤ t``, so
    ``T̃ = e^{−γ Σ_j (t_j − T)} (1 − e^{−mγT})/(mγ)`` with ``T = min_j t_j``.

    Exponential pulses, rates ``r_j = 1/τ_{a_j}``: the ``s < 0`` part is
    ``Π_j A_j(t_j) / Σ_j r_j`` with ``A_j(t) = (e^{−r_j t} − e^{−γ t})/(γ − r_j)``,
    and expanding ``Π_j (e^{−r_j(t_j−s)} − e^{−γ(t_j−s)})`` over subsets ``S``
    of the legs gives the ``0 ≤ s ≤ T`` part,
    ``Σ_S (−1)^{m−|S|} (e^{−E_S} − e^{−c_S}) / (b_S Π_j (γ − r_j))`` with
    ``b_S = Σ_{j∈S} r_j + γ(m − |S|)``, ``c_S = Σ_{j∈S} r_j t_j + γ Σ_{j∉S} t_j``
    and ``E_S = c_S − b_S T``.  Every exponent is non-positive.
    """
    ts = _legs(ts)
    m = ts.shape[0]
    g = p.gamma
    T = ts.min(axis=0)
    positive = T > 0.0
    Tp = np.where(positive, T, 0.0)
    if p.pulse == "white":
        out = (np.exp(-g * (ts - Tp).sum(axis=0))
               * (-np.expm1(-m * g * Tp)) / (m * g))
        return np.where(positive, out, 0.0)
    r = np.array([p.rates[a] for a in comps])
    A = (np.exp(-r[:, None] * ts) - np.exp(-g * ts)) / (g - r)[:, None]
    before = np.prod(A, axis=0) / r.sum()
    after = np.zeros_like(Tp)
    for mask in itertools.product((False, True), repeat=m):
        sel = np.array(mask)
        k = int(sel.sum())
        b = r[sel].sum() + g * (m - k)
        c = (r[sel][:, None] * ts[sel]).sum(axis=0) + g * ts[~sel].sum(axis=0)
        E = c - b * Tp
        after = after + (-1.0) ** (m - k) * (np.exp(-E) - np.exp(-c)) / b
    after = after / np.prod(g - r)
    return np.where(positive, before + after, 0.0)


def K_R(comps, xs, ts, p: Params) -> np.ndarray:
    """The R-contracted cumulant ``ν Π h · X · T̃``, ``(n,)``."""
    amp = p.nu * np.prod([p.h[a] for a in comps])
    return amp * spatial_overlap(comps, xs, p) * t_tilde(comps, ts, p)


def C_matrix(x1, t1, x2, t2, p: Params) -> np.ndarray:
    """The exact C propagator, ``K_R`` at ``m = 2``: ``(n, N, N)``."""
    t1 = np.atleast_1d(np.asarray(t1, float))
    t2 = np.atleast_1d(np.asarray(t2, float))
    n = max(t1.shape[0], t2.shape[0])
    xs = np.stack([np.broadcast_to(np.asarray(x1, float), (n,)),
                   np.broadcast_to(np.asarray(x2, float), (n,))])
    ts = np.stack([np.broadcast_to(t1, (n,)), np.broadcast_to(t2, (n,))])
    N = p.n_components
    out = np.empty((n, N, N))
    for a in range(N):
        for b in range(N):
            out[:, a, b] = K_R((a, b), xs, ts, p)
    return out


# ---------------------------------------------------------------------------
# Inputs of the moment hierarchy (reference.py)
# ---------------------------------------------------------------------------

def jump_moment(counts: dict, positions, p: Params) -> float:
    r"""``ν E[Π J_{(a,i)}^{k}]`` for jump counts ``{(a, i): k}``, where one
    event at ``z`` moves component ``a`` at point ``x_i`` by
    ``h_a w_a(x_i − z)``."""
    comps, xs, powers = [], [], []
    amp = p.nu
    for (a, i), k in counts.items():
        if k == 0:
            continue
        comps.append(a)
        xs.append(positions[i])
        powers.append(k)
        amp *= p.h[a] ** k
    return float(amp * spatial_overlap(comps, np.array(xs, float), p,
                                       powers=powers)[0])


def stationary_eta_cumulant(legs, positions, p: Params) -> float:
    r"""Joint cumulant of ``η_{a_1}(x_{i_1}), …`` at one time, stationary
    exponential pulses: ``ν Π h · X / Σ_j r_{a_j}`` for ``m ≥ 2``, 0 for
    ``m = 1``.  ``legs`` is a sequence of ``(a, i)``."""
    if len(legs) < 2:
        return 0.0
    comps = [a for a, _ in legs]
    xs = np.array([positions[i] for _, i in legs], float)
    amp = p.nu * np.prod([p.h[a] for a in comps])
    return float(amp * spatial_overlap(comps, xs, p)[0]
                 / sum(p.rates[a] for a in comps))
