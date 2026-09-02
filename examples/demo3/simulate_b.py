r"""Level-B simulation: exponential time differencing with exact noise.

The interacting system

    ``dφ_a/dt = −γ φ_a + s·F_abc φ_b φ_c + η_a``

is split as ``φ = φ_free + φ_int``, where ``φ_free`` solves the *linear*
equation driven by the event noise.  ``φ_free`` is available in closed
form at every step, so **the only discretised quantity is the ``F``
term** --- which is itself a small correction --- and its error is
``O(Δt²)`` (exponential Heun / ETDRK2).

Two things make this cheap:

* **A geometric recursion for the free field.**  For ``s_k ≥ 0``,
  ``J(t, s_k) = [e^{−a(t−s_k)} − e^{−γ(t−s_k)}]/(γ−a)``, so the event sum
  factorises into two running sums that update by one multiplication per
  step::

      Ũ_{n+1} = e^{−aΔ} Ũ_n + Σ_{s_k ∈ (t_n, t_{n+1}]} w_k e^{−a(t_{n+1}−s_k)}

  and likewise ``Ṽ`` with ``γ``.  Written this way every exponent is
  non-positive, so it is stable at any ``t`` (the naive ``Σ w e^{+a s_k}``
  overflows).  Cost is O(events) once plus O(1) per step, instead of
  O(events) *per* step.

* **A control variate with an exactly known mean.**  With ``F = 0`` the
  two components are driven by *independent* event processes, so
  ``⟨φ₀^free φ₁^free⟩ = 0`` exactly.  That product is strongly correlated
  with ``φ₀φ₁`` (they differ at O(F)) and has known mean zero, so
  subtracting it removes the whole Gaussian-sector variance --- which
  otherwise dwarfs a signal of ~1e-3 riding on fluctuations of size
  ``κ₂``.  See :func:`control_variate_estimate`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shot_noise import PARAMS, A_response, ShotNoise
from simulate import Window, mean_window

__all__ = ["SimResult", "control_variate_estimate", "simulate_xi"]


@dataclass(frozen=True)
class SimResult:
    """Estimates and batch-scatter errors on a ``(site, time)`` grid.

    ``xi01[j, k]`` is ``⟨φ₀(x₀, t_k) φ₁(x_j, t_k)⟩`` and ``xi00[j, k]`` is
    ``⟨φ₀(x₀, t_k) φ₀(x_j, t_k)⟩`` --- site 0 paired against every site, so
    ``j = 0`` is the coincident case and ``j > 0`` gives the separation
    dependence.
    """

    t_record: np.ndarray
    sites: np.ndarray
    xi01: np.ndarray
    xi01_err: np.ndarray
    xi00: np.ndarray
    xi00_err: np.ndarray
    variance_reduction: np.ndarray
    n_real: int
    dt: float
    blowup_fraction: float


def _weighted(per_batch, sizes):
    """Size-weighted mean of per-batch estimates, and its standard error.

    Batches are **not** assumed equal in size.  Treating batch ``b`` as an
    estimate of variance ``σ²/n_b``, the weighted mean has variance
    ``σ²/N`` with ``σ̂² = Σ n_b (x_b − x̄)² / (B − 1)``; for equal batches
    this reduces to the familiar ``s/√B``.

    Equal weighting is a trap here: a run whose realisation count does not
    divide evenly leaves a small remainder batch, and giving eight
    realisations the same weight as thirteen thousand shifted the answer
    by 20 %.  Every earlier run happened to divide evenly, which is
    exactly why it stayed hidden.
    """
    n_b = np.asarray(sizes, dtype=float).reshape((-1,) + (1,) * (per_batch.ndim - 1))
    total = n_b.sum()
    est = (n_b * per_batch).sum(axis=0) / total
    if per_batch.shape[0] < 2:
        return est, np.zeros_like(est)
    var = (n_b * (per_batch - est) ** 2).sum(axis=0) / (per_batch.shape[0] - 1)
    return est, np.sqrt(var / total)


def control_variate_estimate(m_xy, m_z, m_xz, m_zz, sizes=None):
    """Combine per-batch means into a control-variate estimate.

    ``Ξ = ⟨xy⟩ − c⟨z⟩`` with ``c = Cov(xy, z)/Var(z)`` and ``E[z] = 0``
    known exactly.  ``c`` comes from the *pooled* sample (via the batch
    means of ``xy·z`` and ``z²``), not from a regression over batches, so
    it is not the noisy part; the error bar is the size-weighted scatter
    of the per-batch values of ``xy − c z``, so it reflects the
    *combination* rather than its parts.

    Args:
        m_xy, m_z, m_xz, m_zz: per-batch means of ``xy``, ``z``, ``xy·z``
            and ``z²``, each shaped ``(n_batch, ...)``.
        sizes: per-batch realisation counts; equal sizes assumed if
            omitted.

    Returns:
        ``(estimate, standard_error, variance_reduction_factor)``.
    """
    m_xy, m_z = np.asarray(m_xy), np.asarray(m_z)
    if sizes is None:
        sizes = np.ones(m_xy.shape[0])
    w = np.asarray(sizes, dtype=float).reshape((-1,) + (1,) * (m_xy.ndim - 1))
    tot = w.sum()
    mean_xy = (w * m_xy).sum(axis=0) / tot
    mean_z = (w * m_z).sum(axis=0) / tot
    cov = (w * np.asarray(m_xz)).sum(axis=0) / tot - mean_xy * mean_z
    var = (w * np.asarray(m_zz)).sum(axis=0) / tot - mean_z ** 2
    c = np.where(var > 0, cov / np.where(var > 0, var, 1.0), 0.0)
    est, err = _weighted(m_xy - c * m_z, sizes)
    _, raw_err = _weighted(m_xy, sizes)
    return est, err, np.where(err > 0, raw_err / np.maximum(err, 1e-300), np.inf)


def _draw_free_field(rng, p, sites, window, t_edges, n_batch):
    r"""Exact ``φ_free`` at every step edge, for one component.

    Returns ``(n_batch, n_sites, n_steps + 1)``, already mean-subtracted
    with the **truncated-window** mean so the window contributes no bias.
    """
    n_sites, n_steps = len(sites), len(t_edges) - 1
    mu = p.nu * window.area
    counts = rng.poisson(mu, n_batch)
    total = int(counts.sum())
    xk = rng.uniform(-window.half_length, window.half_length, total)
    sk = rng.uniform(window.s_min, window.s_max, total)
    rep = np.repeat(np.arange(n_batch), counts)

    w = p.h * np.exp(-np.abs(np.asarray(sites)[:, None] - xk[None, :]) / p.sigma_x)

    past = sk < 0.0
    # s < 0: the whole pulse tail is inside [0, t], giving e^{a s} A(t).
    b_const = np.stack([
        np.bincount(rep[past], weights=(w[i][past] * np.exp(p.a * sk[past])),
                    minlength=n_batch) for i in range(n_sites)], axis=1)

    live = ~past
    # Which step interval each event falls in; events at s > t_max are
    # dropped (they cannot influence any recorded time).
    idx = np.searchsorted(t_edges, sk[live], side="left") - 1
    idx = np.clip(idx, 0, n_steps - 1)
    keep = sk[live] <= t_edges[-1]
    ev_idx, b_idx = idx[keep], rep[live][keep]
    tau = t_edges[ev_idx + 1] - sk[live][keep]
    w_live = w[:, live][:, keep]                       # (n_sites, n_ev)
    site = np.arange(n_sites)[:, None]
    flat = (((b_idx[None, :] * n_sites) + site) * n_steps + ev_idx[None, :]).ravel()
    shape = (n_batch, n_sites, n_steps)
    size = n_batch * n_sites * n_steps
    dU = np.bincount(flat, weights=(w_live * np.exp(-p.a * tau)[None, :]).ravel(),
                     minlength=size).reshape(shape)
    dV = np.bincount(flat, weights=(w_live * np.exp(-p.gamma * tau)[None, :]).ravel(),
                     minlength=size).reshape(shape)

    dt = t_edges[1] - t_edges[0]
    decay_a, decay_g = np.exp(-p.a * dt), np.exp(-p.gamma * dt)
    phi = np.empty((n_batch, n_sites, n_steps + 1))
    U = np.zeros((n_batch, n_sites))
    V = np.zeros((n_batch, n_sites))
    phi[:, :, 0] = 0.0
    for n in range(n_steps):
        U = decay_a * U + dU[:, :, n]
        V = decay_g * V + dV[:, :, n]
        phi[:, :, n + 1] = (U - V) / (p.gamma - p.a)
    phi += A_response(t_edges, p)[None, None, :] * b_const[:, :, None]
    phi *= p.h
    mean = np.stack([mean_window(np.full_like(t_edges, x), t_edges, window, p,
                                 kind="phi") for x in sites], axis=0)
    return phi - mean[None, :, :]


def _f_term(phi0, phi1, s):
    """``s·F_abc φ_b φ_c`` for the Z₂ tensor: ``g₀ = s φ₁²``, ``g₁ = s φ₀φ₁``."""
    return s * phi1 * phi1, s * phi0 * phi1


def _step_trajectories(free0, free1, s, gamma, dt):
    """Integrate the interaction on top of the exact free field (ETDRK2).

    ``φ = φ_free + φ_int``; only ``φ_int`` is stepped, so the linear part
    and the noise stay exact and the discretisation error is ``O(Δt²)`` on
    a term that is itself a small correction.  Over one step

        ``∫ e^{−γ(Δ−u)} g(u) du ≈ (φ₁−φ₂) g_n + φ₂ g*``,
        ``φ₁ = (1−e^{−γΔ})/γ``,  ``φ₂ = 1/γ − (1−e^{−γΔ})/(γ²Δ)``,

    with ``g*`` from an exponential-Euler predictor.

    Args:
        free0, free1: ``(n_real, n_sites, n_steps + 1)`` exact free fields.
        s: ``F`` amplitude.  gamma, dt: drift rate and step.

    Returns:
        the two full fields, same shape as the inputs.
    """
    e_dt = np.exp(-gamma * dt)
    c1 = (1.0 - e_dt) / gamma
    c2 = 1.0 / gamma - (1.0 - e_dt) / (gamma ** 2 * dt)
    n_steps = free0.shape[2] - 1
    int0 = np.zeros_like(free0[:, :, 0])
    int1 = np.zeros_like(int0)
    full0, full1 = np.empty_like(free0), np.empty_like(free1)
    full0[:, :, 0], full1[:, :, 0] = free0[:, :, 0], free1[:, :, 0]
    for n in range(n_steps):
        g0, g1 = _f_term(full0[:, :, n], full1[:, :, n], s)
        star0 = e_dt * int0 + c1 * g0
        star1 = e_dt * int1 + c1 * g1
        gs0, gs1 = _f_term(free0[:, :, n + 1] + star0,
                           free1[:, :, n + 1] + star1, s)
        int0 = e_dt * int0 + (c1 - c2) * g0 + c2 * gs0
        int1 = e_dt * int1 + (c1 - c2) * g1 + c2 * gs1
        full0[:, :, n + 1] = free0[:, :, n + 1] + int0
        full1[:, :, n + 1] = free1[:, :, n + 1] + int1
    return full0, full1


def simulate_xi(p: ShotNoise, f_amplitude: float, sites, t_record,
                n_real: int, dt: float = 0.01, seed: int = 0,
                batch: int = 20_000, window: Window | None = None,
                blowup_threshold: float = 1e3, n_err_batches: int = 25):
    """Simulate ``ξ₀₁`` and ``ξ₀₀`` on a ``(site, time)`` grid.

    Args:
        p: model parameters.
        f_amplitude: ``s``; the observable's leading channel is ``∝ s``.
        sites: site positions (placed exactly at the separations wanted).
        t_record: times at which to record; snapped to the step grid.
        n_real: total realisations.
        dt: step size for the ``F`` term only --- the linear part and the
            noise are integrated exactly.
        blowup_threshold: |φ| above which a trajectory is counted as
            diverged (the quadratic drift can run away in finite time).
        n_err_batches: minimum number of chunks the run is split into, so
            the batch-scatter error bar is itself well determined --- with
            only two or three chunks the "error" is noise and a perfectly
            good estimate can look like a 16-sigma discrepancy.

    Returns:
        :class:`SimResult`.
    """
    sites = np.atleast_1d(np.asarray(sites, dtype=float))
    t_record = np.atleast_1d(np.asarray(t_record, dtype=float))
    t_max = float(t_record.max())
    n_steps = int(np.ceil(t_max / dt))
    t_edges = np.linspace(0.0, n_steps * dt, n_steps + 1)
    rec_idx = np.searchsorted(t_edges, t_record)
    rec_idx = np.clip(rec_idx, 0, n_steps)
    if window is None:
        window = Window.for_times(t_edges[-1], p)
    batch = max(1, min(batch, n_real // max(n_err_batches, 1)))

    acc = {k: [] for k in ("xy", "z", "xz", "zz", "xx")}
    sizes = []
    n_blow, n_done = 0, 0
    rng = np.random.default_rng(seed)
    while n_done < n_real:
        nb = min(batch, n_real - n_done)
        free0 = _draw_free_field(rng, p, sites, window, t_edges, nb)
        free1 = _draw_free_field(rng, p, sites, window, t_edges, nb)
        full0, full1 = _step_trajectories(free0, free1, f_amplitude,
                                          p.gamma, dt)
        f0, f1 = full0[:, :, rec_idx], full1[:, :, rec_idx]
        z0, z1 = free0[:, :, rec_idx], free1[:, :, rec_idx]
        bad = (np.abs(f0).max(axis=(1, 2)) > blowup_threshold) | \
              (np.abs(f1).max(axis=(1, 2)) > blowup_threshold)
        n_blow += int(bad.sum())
        ok = ~bad
        # Site 0 of component 0 against every site of component 1, so the
        # same call serves xi(t) at r = 0 (one site) and xi(r) at fixed t
        # (several).  The sites are placed exactly at the separations to
        # be plotted, so nothing is ever interpolated.
        xy = f0[ok][:, :1, :] * f1[ok]
        z = z0[ok][:, :1, :] * z1[ok]
        acc["xy"].append(xy.mean(axis=0))
        acc["z"].append(z.mean(axis=0))
        acc["xz"].append((xy * z).mean(axis=0))
        acc["zz"].append((z * z).mean(axis=0))
        acc["xx"].append((f0[ok][:, :1, :] * f0[ok]).mean(axis=0))
        sizes.append(int(ok.sum()))
        n_done += nb

    xi01, err01, vr = control_variate_estimate(
        acc["xy"], acc["z"], acc["xz"], acc["zz"], sizes)
    xi00, err00 = _weighted(np.asarray(acc["xx"]), sizes)
    return SimResult(t_record=t_edges[rec_idx], sites=sites, xi01=xi01,
                     xi01_err=err01, xi00=xi00, xi00_err=err00,
                     variance_reduction=vr, n_real=n_real, dt=dt,
                     blowup_fraction=n_blow / n_real)
