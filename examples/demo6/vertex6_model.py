r"""Demo 6's model: static and white non-local vertices, cubic and quartic
drift.  Parameters and closed forms; numpy only.

The field obeys, for ``t ≥ t_min`` from ``φ(t_min) = 0``,

.. math::

    dφ_a = \bigl(−γ_a φ_a + F_{abc} φ_b φ_c + G_{abcd} φ_b φ_c φ_d
                 + X_a + η_a\bigr)\,dt + dW_a ,

with four sources of noise, each entered in the package differently:

* ``dW``: Gaussian white noise, correlated in space and across components,
  ``⟨dW_a(x) dW_b(y)⟩ = σ²_ab(x, y) dt`` with
  ``σ²_ab(x, y) = Q_ab ∫dz w_a(x−z) w_b(y−z)``,
  ``w_a(u) = e^{−u²/(2ℓ_a²)}``.  It gives the C propagator.
* ``X``: a random vector constant in space and time (``dX = 0``) with
  cumulants ``κ^(m)`` (``m = 2 … 5``).  Its cumulant is the same at any
  set of points and times, which is the package's *static* non-local
  vertex (an ndarray coupling, ``equal_time=False``).
* ``η``: white compound-Poisson jumps whose jump vector does not depend on
  the point, ``η_a(t) = Σ_k J^k_a δ(t − s_k)`` (compensated).  Campbell's
  theorem gives the cumulants ``ν E[J^{⊗m}] δ(t_1−t_2)…δ(t_{m−1}−t_m)``,
  the package's static ``equal_time`` vertex.  At ``m = 2`` the same
  structure is a Gaussian white noise entered as a vertex.

With ``F = G = 0`` every cumulant of ``φ`` is closed form: a static
``κ^(m)`` contributes ``κ^(m)_{a} Π_j g_{a_j}(t_j)`` with
``g_a(t) = (1 − e^{−γ_a (t − t_min)})/γ_a``, an equal-time one
``κ_eq^(m)_{a} ∫_{t_min}^{min t} ds Π_j e^{−γ_{a_j}(t_j − s)}``.

Every cumulant tensor is symmetric (a cumulant of a random vector is), with
distinct entries of both signs; ``F`` and ``G`` have no index symmetry;
the rates, widths and amplitudes differ between the components.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, replace

import numpy as np

__all__ = ["Params", "PARAMS", "PARAMS_COMMON", "F_TENSOR", "G_TENSOR",
           "X_CUMULANTS", "JUMP_CUMULANTS", "symmetric_tensor", "g_factor",
           "equal_time_block", "sigma2", "C_matrix", "split_sum",
           "set_partitions"]

N = 2


@dataclass(frozen=True)
class Params:
    """Model parameters.

    Args:
        rates: ``γ_a``.  Distinct rates make R a (diagonal) matrix.
        t_min: start time; ``φ(t_min) = 0``.
        Q: amplitude matrix of the white noise (positive definite, mixing).
        ell: spatial width of each component's white noise.
    """

    rates: tuple = (0.8, 1.3)
    t_min: float = 0.4
    Q: tuple = ((0.50, -0.20), (-0.20, 0.35))
    ell: tuple = (0.7, 1.2)

    @property
    def n_components(self) -> int:
        return len(self.rates)

    @property
    def common_rate(self) -> bool:
        return bool(np.allclose(self.rates, self.rates[0]))


#: Distinct rates: a diagonal matrix R (the scalar loops only, today).
PARAMS = Params()
#: One rate for both components: a scalar R (every integrator).
PARAMS_COMMON = replace(PARAMS, rates=(1.05, 1.05))


def symmetric_tensor(values, m: int) -> np.ndarray:
    """The symmetric ``(N,)*m`` tensor whose entry depends on the number of
    indices equal to 1: ``values[k]`` for ``k`` ones (``N = 2``)."""
    if len(values) != m + 1:
        raise ValueError(f"need {m + 1} values for m = {m}")
    K = np.empty((N,) * m)
    for idx in itertools.product(range(N), repeat=m):
        K[idx] = values[sum(idx)]
    return K


#: ``dφ_a ⊃ F_abc φ_b φ_c dt``; no index symmetry.
F_TENSOR = np.array([[[0.30, -0.45], [0.20, 0.15]],
                     [[-0.25, 0.10], [0.40, -0.35]]])

#: ``dφ_a ⊃ G_abcd φ_b φ_c φ_d dt``; no index symmetry.
G_TENSOR = np.array([[[[0.12, -0.30], [0.25, 0.05]],
                      [[-0.18, 0.22], [0.08, -0.27]]],
                     [[[0.20, 0.14], [-0.33, 0.11]],
                      [[0.06, -0.24], [0.17, -0.09]]]])

#: Cumulants of the static force ``X`` (``κ^(2)`` is its covariance).
X_CUMULANTS = {
    2: symmetric_tensor((0.60, -0.25, 0.45), 2),
    3: symmetric_tensor((0.80, -0.35, 0.50, -0.60), 3),
    4: symmetric_tensor((0.90, -0.30, 0.45, 0.20, -0.70), 4),
    5: symmetric_tensor((0.60, 0.25, -0.40, 0.35, -0.20, 0.50), 5),
}

#: Rate-weighted jump moments ``ν E[J^{⊗m}]`` of the white jumps; at
#: ``m = 2`` the covariance rate of a Gaussian white noise entered as a
#: vertex.
JUMP_CUMULANTS = {
    2: symmetric_tensor((0.40, 0.15, 0.30), 2),
    3: symmetric_tensor((-0.50, 0.30, 0.70, -0.40), 3),
    4: symmetric_tensor((0.55, -0.20, 0.35, -0.45, 0.25), 4),
    5: symmetric_tensor((-0.30, 0.45, 0.20, -0.50, 0.40, -0.25), 5),
}


# ---------------------------------------------------------------------------
# Closed forms
# ---------------------------------------------------------------------------

def g_factor(a: int, t, p: Params):
    """``∫_{t_min}^t e^{−γ_a (t − s)} ds = (1 − e^{−γ_a (t − t_min)})/γ_a``."""
    t = np.asarray(t, dtype=float)
    dt = np.maximum(t - p.t_min, 0.0)
    return -np.expm1(-p.rates[a] * dt) / p.rates[a]


def equal_time_block(comps, ts, p: Params):
    r"""``∫_{t_min}^{min t} ds Π_j e^{−γ_{a_j}(t_j − s)}``, ``ts`` of shape
    ``(m,)`` or ``(m, n)``."""
    ts = np.asarray(ts, dtype=float)
    ts2 = ts[:, None] if ts.ndim == 1 else ts
    r = np.array([p.rates[a] for a in comps])[:, None]
    m = ts2.min(axis=0)
    span = np.maximum(m - p.t_min, 0.0)
    total = r.sum()
    out = (np.exp(-(r * (ts2 - m)).sum(axis=0))
           * (-np.expm1(-total * span)) / total)
    return out[0] if ts.ndim == 1 else out


def sigma2(a: int, x, b: int, y, p: Params):
    r"""``σ²_ab(x, y) = Q_ab \sqrt{2π ℓ_a² ℓ_b²/(ℓ_a²+ℓ_b²)}
    e^{−(x−y)²/(2(ℓ_a²+ℓ_b²))}``."""
    la2, lb2 = p.ell[a] ** 2, p.ell[b] ** 2
    s = la2 + lb2
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    return p.Q[a][b] * np.sqrt(2.0 * np.pi * la2 * lb2 / s) * np.exp(
        -0.5 * d * d / s)


def C_matrix(x1, t1, x2, t2, p: Params) -> np.ndarray:
    """The exact C, ``σ²_ab(x1, x2) ∫ e^{−γ_a(t1−s)} e^{−γ_b(t2−s)} ds``:
    ``(n, N, N)``."""
    t1 = np.atleast_1d(np.asarray(t1, dtype=float))
    t2 = np.atleast_1d(np.asarray(t2, dtype=float))
    n = max(t1.shape[0], t2.shape[0])
    t1 = np.broadcast_to(t1, (n,))
    t2 = np.broadcast_to(t2, (n,))
    x1 = np.broadcast_to(np.asarray(x1, dtype=float), (n,))
    x2 = np.broadcast_to(np.asarray(x2, dtype=float), (n,))
    Nc = p.n_components
    out = np.empty((n, Nc, Nc))
    ts = np.stack([t1, t2])
    for a in range(Nc):
        for b in range(Nc):
            out[:, a, b] = (sigma2(a, x1, b, x2, p)
                            * equal_time_block((a, b), ts, p))
    return out


def set_partitions(items: list):
    """Every partition of ``items`` into non-empty blocks."""
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in set_partitions(rest):
        for k in range(len(part)):
            yield part[:k] + [[first] + part[k]] + part[k + 1:]
        yield [[first]] + part


def split_sum(comps, ts, p: Params, block_sizes, *, equal_time: bool,
              tensors) -> float:
    r"""``Σ`` over partitions of the legs into blocks with the given sizes of
    ``Π_B κ^{(|B|)}_{a_B} w_B``, where ``w_B = Π_{j∈B} g_{a_j}(t_j)`` for a
    static cumulant and the equal-time block integral otherwise.

    This is the part of ``E[Π_j φ_{a_j}(t_j)]`` at ``F = G = 0`` that is a
    product of exactly these cumulants.
    """
    want = sorted(block_sizes)
    total = 0.0
    for part in set_partitions(list(range(len(comps)))):
        if sorted(len(b) for b in part) != want:
            continue
        term = 1.0
        for block in part:
            bc = tuple(comps[j] for j in block)
            term *= tensors[len(block)][bc]
            if equal_time:
                term *= equal_time_block(bc, [ts[j] for j in block], p)
            else:
                term *= np.prod([g_factor(comps[j], ts[j], p) for j in block])
        total += term
    return float(total)
