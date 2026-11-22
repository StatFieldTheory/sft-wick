"""Deterministic, variance-scaled probes of candidate C interpolators."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import isfinite
from numbers import Real

import numpy as np


DEFAULT_TABLE_RTOL = 1e-6
C_TABLE_CERTIFICATE_VERSION = 2
_PROBE_FACTOR = 1.25


def validate_table_rtol(value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("table_rtol must be a finite positive number or None")
    value = float(value)
    if not isfinite(value) or value <= 0:
        raise ValueError("table_rtol must be a finite positive number or None")
    return value


@dataclass(frozen=True)
class CTableError:
    """Worst sampled normalized residual for one probe set."""

    normalized: float
    absolute: float
    scale: float
    component: tuple[int, int]
    times: tuple[float, float]
    positions: tuple[tuple[float, ...], tuple[float, ...]]


@dataclass(frozen=True)
class CTableCertificate:
    """A posteriori measured C error in units of component covariance."""

    tolerance: float
    time: CTableError
    nodes: CTableError
    space: CTableError | None = None
    joint: CTableError | None = None

    @property
    def estimate(self) -> float:
        errors = [self.time.normalized, self.nodes.normalized]
        if self.space is not None:
            errors.append(self.space.normalized)
        if self.joint is not None:
            errors.append(self.joint.normalized)
        return _PROBE_FACTOR * max(errors)


class CTableAccuracyError(ValueError):
    """The candidate C table exceeded its measured error tolerance."""

    def __init__(self, certificate: CTableCertificate):
        self.certificate = certificate
        categories = {"time": certificate.time, "node integral": certificate.nodes}
        if certificate.space is not None:
            categories["space"] = certificate.space
        if certificate.joint is not None:
            categories["joint time and space"] = certificate.joint
        kind, worst = max(categories.items(), key=lambda item: item[1].normalized)
        super().__init__(
            f"C table failed table_rtol={certificate.tolerance:g} "
            f"(sampled estimate {certificate.estimate:.6g}, worst {kind} residual). "
            f"C{worst.component} at times {worst.times} and positions {worst.positions}: "
            f"absolute error {worst.absolute:.6g}, scale {worst.scale:.6g}. "
            "Refine the time grid, spatial grid or node quadrature indicated "
            "by the failing category, or use a direct closed form."
        )


def _temporal_probes(ts):
    """Probe a physical min-time/lag triangle in both orientations."""
    span = float(ts[-1] - ts[0])
    count = len(ts)
    step = span / (count - 1)
    cells = {0, 1, 3, (count - 2) // 3, (2 * (count - 2)) // 3,
             count - 3, count - 2}
    points = {(0.0, 0.0), (0.0, span), (span, 0.0), (0.0, 0.37 * span)}
    for cell in sorted(cells):
        if not 0 <= cell < count - 1:
            continue
        for offset in (0.21, 0.50):
            time = (cell + offset) * step
            available = span - time
            for lag in (
                0.0, 0.20 * step, 0.50 * step, 0.24 * available, 0.62 * available,
                max(0.0, available - 0.29 * step), available,
            ):
                if lag <= available:
                    points.add((time, lag))
    points = np.array(sorted(points))
    earlier = ts[0] + points[:, 0]
    later = np.minimum(ts[-1], ts[0] + (points[:, 0] + points[:, 1]))
    return np.r_[later, earlier], np.r_[earlier, later]


def _node_probes(ts):
    indices = np.unique([0, 1, len(ts) // 2, len(ts) - 2, len(ts) - 1])
    nodes = ts[indices]
    return np.repeat(nodes, len(nodes)), np.tile(nodes, len(nodes))


def _joint_time_probes(ts):
    cells = np.unique([0, (len(ts) - 2) // 2, len(ts) - 2])
    midpoint = 0.5 * ts[cells] + 0.5 * ts[cells + 1]
    late = np.full_like(midpoint, ts[-1])
    return np.r_[midpoint, midpoint, late], np.r_[midpoint, late, midpoint]


def spatial_probes(axes):
    """Sample knots, single-axis offsets and joint interior displacements."""
    indices = [np.unique([0, len(axis) // 2, len(axis) - 1]) for axis in axes]
    knots = np.array(np.meshgrid(*[axis[i] for i, axis in zip(indices, axes)])).reshape(
        len(axes), -1
    ).T
    off_knots = set()
    midpoints = []
    for dimension, axis in enumerate(axes):
        if len(axis) < 2:
            midpoints.append((axis[0],))
            continue
        cells = sorted({0, (len(axis) - 2) // 2, len(axis) - 2})
        midpoints.append(tuple(0.5 * axis[cell] + 0.5 * axis[cell + 1] for cell in cells))
        for cell in cells:
            for offset in (0.29, 0.50, 0.71):
                location = axis[cell] + offset * (axis[cell + 1] - axis[cell])
                for fixed in (axis[0], axis[-1]):
                    sample = [fixed] * len(axes)
                    sample[dimension] = location
                    off_knots.add(tuple(sample))
    # A covariance can vanish on every spatial edge while remaining nonzero
    # in a cell interior. Probe all midpoint combinations of the chosen cells.
    joint = list(product(*midpoints)) if off_knots else []
    off_knots.update(joint)
    return knots, sorted(off_knots), joint


def measure_table(tolerance, ts, entries, reference, samples):
    """Measure the candidate at physical time probes and sampled spatial axes.

    Samples contain lists of (x1, x2, lookup) triples for knots, off-knot
    space and simultaneous space-time offsets.
    """
    knots, off_knots, joint = samples
    time_probes, node_probes = _temporal_probes(ts), _node_probes(ts)
    anchors = np.linspace(ts[0], ts[-1], 5)
    variances = {}

    def variance(x):
        key = tuple(np.asarray(x).ravel())
        if key not in variances:
            matrices = reference(x, anchors, x, anchors)
            diagonal = np.diagonal(matrices, axis1=1, axis2=2)
            variances[key] = np.max(diagonal, axis=0)
        return variances[key]

    components_a, components_b = np.array(entries).T

    def worst(probes, x1, x2, lookup):
        t1, t2 = probes
        direct = reference(x1, t1, x2, t2)[:, components_a, components_b]
        scale = np.sqrt(np.maximum(variance(x1)[components_a], 0.0))
        scale *= np.sqrt(np.maximum(variance(x2)[components_b], 0.0))
        residual = np.abs(lookup(t1, t2) - direct)
        normalized = np.zeros_like(residual)
        np.divide(residual, scale[None, :], where=scale[None, :] > 0, out=normalized)
        normalized[(scale[None, :] <= 0) & (residual != 0)] = np.inf
        normalized[~np.isfinite(residual) | ~np.isfinite(scale[None, :])] = np.inf
        row, col = np.unravel_index(np.argmax(normalized), residual.shape)
        return CTableError(
            float(normalized[row, col]),
            float(residual[row, col]),
            float(scale[col]),
            entries[col],
            (float(t1[row]), float(t2[row])),
            (tuple(map(float, np.ravel(x1))), tuple(map(float, np.ravel(x2)))),
        )

    def worst_sample(probes, positions):
        return max((worst(probes, *sample) for sample in positions),
                   key=lambda record: record.normalized)

    return CTableCertificate(
        tolerance,
        time=worst_sample(time_probes, knots),
        nodes=worst_sample(node_probes, knots),
        space=worst_sample(node_probes, off_knots) if off_knots else None,
        joint=worst_sample(_joint_time_probes(ts), joint) if joint else None,
    )
