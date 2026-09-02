r"""Event-exact simulation of the filtered-Poisson model.

The point of demo 3 is that the reference simulation has **no
discretisation error at all** in the ``F = 0`` case.  Two facts conspire:

* **No spatial discretisation.**  The drift ``−γφ + Fφφ`` carries no
  spatial derivative, so distinct sites are coupled *only* through the
  noise.  A finite set of sites is therefore an exact realisation of the
  model at those sites, and the sites can be placed exactly at the
  separations to be plotted --- no interpolation (the off-grid
  interpolation is what biases demo 2's ``r`` by +0.5 %).

* **No time stepping.**  With ``F = 0`` the response of the linear system
  to a single exponential pulse is analytic::

      φ_a(x, t) = Σ_k h · w(x − x_k) · J(t, s_k)   −   ⟨·⟩,

  with ``J`` from :mod:`shot_noise`.  Each event contributes in closed
  form, so a realisation is exact and the *only* error is Monte Carlo.

The single approximation is truncating the event window to
``|x| ≤ L`` and ``s ≥ s_min``.  It is bounded analytically by
:func:`truncation_bound`: because the ``m``-th cumulant is an ``m``-fold
product of pulse profiles, its relative truncation error is
``~e^{−m L/σ_x}`` (resp. ``e^{−m |s_min|/σ_t}``) --- ``m`` times faster
than the field's own tail.  The *mean* that is subtracted is computed for
the **same truncated window** (:func:`mean_window`), so the truncation
contributes no bias to it at all.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shot_noise import PARAMS, A_response, J, ShotNoise

__all__ = ["Window", "truncation_bound", "mean_window", "sample_points",
           "central_moments", "connected_cumulant"]


@dataclass(frozen=True)
class Window:
    """Truncated event window: ``x ∈ [−L, L]``, ``s ∈ [s_min, s_max]``.

    Args:
        half_length: ``L`` in units of length (not of ``σ_x``).
        s_min: earliest event time; must be < 0 so the noise is
            (approximately) stationary by ``t = 0``.
        s_max: latest event time; must be ≥ every observation time.
    """

    half_length: float
    s_min: float
    s_max: float

    @property
    def area(self) -> float:
        return 2.0 * self.half_length * (self.s_max - self.s_min)

    @classmethod
    def for_times(cls, t_max: float, p: ShotNoise = PARAMS,
                  n_sigma_x: float = 8.0, n_sigma_t: float = 8.0) -> "Window":
        """A window covering ``n_sigma`` decay lengths around the data."""
        return cls(half_length=n_sigma_x * p.sigma_x,
                   s_min=-n_sigma_t * p.sigma_t,
                   s_max=float(t_max))


def truncation_bound(window: Window, m: int, p: ShotNoise = PARAMS,
                     x_max: float = 0.0) -> dict[str, float]:
    """Relative bound on the ``m``-th cumulant from truncating the window.

    ``X_m`` and ``T_m`` are ``m``-fold products of the pulse profiles, so
    the discarded region contributes a fraction ``e^{−m·L'/σ_x}``
    (``L' = L − max|x_i|``) spatially and ``e^{−m|s_min|/σ_t}``
    temporally.  Both are quoted; the total is their sum.
    """
    lx = max(window.half_length - abs(x_max), 0.0)
    spatial = float(np.exp(-m * lx / p.sigma_x))
    temporal = float(np.exp(-m * abs(window.s_min) / p.sigma_t))
    return {"spatial": spatial, "temporal": temporal,
            "total": spatial + temporal}


def _spatial_window_integral(x, window: Window, p: ShotNoise):
    r"""``∫_{−L}^{L} e^{−|x−x'|/σ_x} dx'`` for ``|x| ≤ L``."""
    x = np.asarray(x, dtype=float)
    L, sx = window.half_length, p.sigma_x
    return sx * (2.0 - np.exp(-(L - x) / sx) - np.exp(-(L + x) / sx))


def mean_window(xs, ts, window: Window, p: ShotNoise = PARAMS,
                kind: str = "phi"):
    r"""Exact mean of the **truncated-window** event sum.

    Subtracting this (rather than the infinite-window mean) makes the
    simulated field exactly centred for the process actually simulated,
    so window truncation contributes no bias to the estimated cumulants.

    ``kind='phi'`` uses ``∫_{s_min}^{t} J(t,s) ds = σ_t(1−e^{−γt})/γ
    − A(t) e^{a s_min}/a`` (both terms stable); ``kind='eta'`` uses
    ``∫ g(t−s) ds = σ_t [e^{−(t−s_hi)/σ_t} − e^{−(t−s_min)/σ_t}]``.
    """
    xs = np.asarray(xs, dtype=float)
    ts = np.asarray(ts, dtype=float)
    spatial = _spatial_window_integral(xs, window, p)
    if kind == "phi":
        if window.s_max < ts.max():
            raise ValueError("window.s_max must cover every observation time")
        temporal = (p.sigma_t * (1.0 - np.exp(-p.gamma * ts)) / p.gamma
                    - A_response(ts, p) * np.exp(p.a * window.s_min) / p.a)
    elif kind == "eta":
        s_hi = np.minimum(ts, window.s_max)
        temporal = p.sigma_t * (np.exp(-(ts - s_hi) / p.sigma_t)
                                - np.exp(-(ts - window.s_min) / p.sigma_t))
    else:
        raise ValueError(f"kind must be 'phi' or 'eta', got {kind!r}")
    return p.nu * p.h * spatial * temporal


def _kernel(kind: str, ts, sk, p: ShotNoise):
    if kind == "phi":
        return J(ts, sk, p)
    tau = ts - sk
    return np.where(tau >= 0.0, np.exp(-np.maximum(tau, 0.0) / p.sigma_t), 0.0)


def sample_points(rng, xs, ts, n_real: int, window: Window,
                  p: ShotNoise = PARAMS, kind: str = "phi",
                  n_fields: int = 1, batch: int = 20_000):
    """Draw ``n_real`` independent realisations of the field at ``m`` points.

    Args:
        rng: a ``numpy.random.Generator``.
        xs, ts: length-``m`` observation positions / times.
        n_real: number of independent realisations.
        window: truncated event window.
        p: model parameters.
        kind: ``'phi'`` (the ``F = 0`` field, exact) or ``'eta'`` (the
            driving noise itself).
        n_fields: number of independent components to draw (each an
            independent copy of the event process).
        batch: realisations per chunk, to bound peak memory.

    Returns:
        ``(n_real, n_fields, m)`` array, already mean-subtracted.
    """
    xs = np.atleast_1d(np.asarray(xs, dtype=float))
    ts = np.atleast_1d(np.asarray(ts, dtype=float))
    m = xs.size
    mu = p.nu * window.area
    mean = mean_window(xs, ts, window, p, kind)
    out = np.empty((n_real, n_fields, m))
    done = 0
    while done < n_real:
        nb = min(batch, n_real - done)
        for f in range(n_fields):
            counts = rng.poisson(mu, nb)
            total = int(counts.sum())
            xk = rng.uniform(-window.half_length, window.half_length, total)
            sk = rng.uniform(window.s_min, window.s_max, total)
            rep = np.repeat(np.arange(nb), counts)
            for j in range(m):
                wgt = (p.h * np.exp(-np.abs(xs[j] - xk) / p.sigma_x)
                       * _kernel(kind, ts[j], sk, p))
                out[done:done + nb, f, j] = np.bincount(rep, weights=wgt, minlength=nb)
        done += nb
    return out - mean[None, None, :]


def central_moments(samples, order: int, n_batch: int = 20):
    """Mean and batch-scatter standard error of ``⟨Π_j x_j⟩`` over realisations.

    Args:
        samples: ``(n_real, m)`` already-centred values; the product is
            taken over the ``m`` axis, so ``order`` is just ``m``.
        order: expected ``m``; checked.
        n_batch: number of batches used for the standard error.

    Returns:
        ``(estimate, standard_error)``.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.shape[1] != order:
        raise ValueError(f"expected {order} points, got {samples.shape[1]}")
    prod = np.prod(samples, axis=1)
    means = np.array([b.mean() for b in np.array_split(prod, n_batch)])
    return float(prod.mean()), float(means.std(ddof=1) / np.sqrt(n_batch))


def connected_cumulant(samples, n_batch: int = 20):
    """Connected 4-point cumulant of centred samples, with a batch standard error.

    ``κ₄(1,2,3,4) = ⟨x₁x₂x₃x₄⟩ − ⟨x₁x₂⟩⟨x₃x₄⟩ − ⟨x₁x₃⟩⟨x₂x₄⟩
    − ⟨x₁x₄⟩⟨x₂x₃⟩``; every piece is estimated from the same batch so the
    scatter of the per-batch estimates is an honest error bar on the
    *combination*, not on its parts.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.shape[1] != 4:
        raise ValueError("connected_cumulant expects 4 points")

    def _one(b):
        f = b[:, 0] * b[:, 1] * b[:, 2] * b[:, 3]
        p12, p34 = (b[:, 0] * b[:, 1]).mean(), (b[:, 2] * b[:, 3]).mean()
        p13, p24 = (b[:, 0] * b[:, 2]).mean(), (b[:, 1] * b[:, 3]).mean()
        p14, p23 = (b[:, 0] * b[:, 3]).mean(), (b[:, 1] * b[:, 2]).mean()
        return f.mean() - p12 * p34 - p13 * p24 - p14 * p23

    per = np.array([_one(b) for b in np.array_split(samples, n_batch)])
    return float(per.mean()), float(per.std(ddof=1) / np.sqrt(len(per)))
