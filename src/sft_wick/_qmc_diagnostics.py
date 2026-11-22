"""Concentration diagnostics for independent QMC scrambles."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class _WeightMoments(NamedTuple):
    peak: float
    scaled_sum: float
    scaled_squares: float


class _TailDiagnostics(NamedTuple):
    max_weight_share: float
    effective_samples: float
    scramble_skew: float | None


def _validate_qmc_heavy_tail(value, replicas) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError("qmc_heavy_tail must be a boolean.")
    if value and replicas is None:
        raise ValueError("qmc_heavy_tail requires qmc_replicas.")
    return bool(value)


def _weight_moments(values: np.ndarray) -> _WeightMoments:
    """Summarize one scramble's absolute, Jacobian-weighted contributions."""
    weights = np.abs(values)
    peak = float(np.max(weights))
    if not np.isfinite(peak):
        raise ValueError("QMC weight diagnostics require finite sample contributions.")
    if peak == 0.0:
        return _WeightMoments(0.0, 0.0, 0.0)
    weights /= peak
    return _WeightMoments(peak, float(weights.sum()), float(np.square(weights).sum()))


def _scramble_skew(estimates: tuple[float, ...]) -> float | None:
    """Adjusted sample skewness of the signed replica means."""
    n = len(estimates)
    if n < 3:
        return None
    samples = np.asarray(estimates, dtype=float)
    _, exponent = np.frexp(np.max(np.abs(samples)))
    scaled = np.ldexp(samples, -exponent)
    shifted = scaled - scaled[0]
    centred = shifted - shifted.mean()
    amplitude = float(np.max(np.abs(centred)))
    if amplitude == 0.0:
        return None
    centred /= amplitude
    return float(
        np.sqrt(n * (n - 1)) / (n - 2)
        * np.mean(centred**3) / np.mean(centred**2)**1.5
    )


def _tail_diagnostics(
    moments: list[_WeightMoments], estimates: tuple[float, ...],
) -> _TailDiagnostics | None:
    peak = max((m.peak for m in moments), default=0.0)
    if peak == 0.0:
        return None
    relative = [m.peak / peak for m in moments]
    total = sum(f * m.scaled_sum for f, m in zip(relative, moments))
    squares = sum(f * f * m.scaled_squares for f, m in zip(relative, moments))
    return _TailDiagnostics(1.0 / total, total * total / squares, _scramble_skew(estimates))


def _is_heavy_tail(metrics: _TailDiagnostics) -> bool:
    """Refuse the SE when very few observed samples carry the estimate."""
    return metrics.max_weight_share >= 0.5 or metrics.effective_samples <= 4.0
