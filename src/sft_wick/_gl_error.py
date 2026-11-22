"""Arithmetic budgets and data for opt-in GL error estimates."""

from __future__ import annotations

from math import isfinite, nextafter, ulp
from typing import NamedTuple

import numpy as np


_UNIT_ROUNDOFF = ulp(1.0) / 2
_MIN_SUBNORMAL = ulp(0.0)
_MIN_NORMAL = np.finfo(float).tiny


def _roundoff_budget(operations: int, magnitude: float) -> float:
    """Standard gamma_k model for products and sums of represented inputs."""
    ku = operations * _UNIT_ROUNDOFF
    if ku >= 1.0 or not isfinite(magnitude):
        raise ValueError("Gauss-Legendre arithmetic budget is not finite.")
    budget = ku / (1.0 - ku) * magnitude
    return nextafter(budget, float("inf")) if ku and magnitude else 0.0


def _add_nonnegative(left: float, right: float) -> float:
    total = left + right
    return nextafter(total, float("inf")) if left and right else total


class _GLRoundoff:
    """Observe absolute weighted sums as each quadrature block is reduced."""

    def __init__(self):
        self.count = 0
        self.norm = 0.0
        self.block_budget = 0.0

    def observe(
        self, weighted: np.ndarray, dimensions: int, unweighted: np.ndarray,
    ) -> None:
        # Include the weight and Jacobian products, their two multiplications
        # into the sample and a sequential upper count for adding the nodes.
        count = len(weighted)
        operations = count + 2 * dimensions - 1
        absolute = np.abs(weighted)
        norm = float(np.sum(absolute))
        norm = _add_nonnegative(norm, _roundoff_budget(count - 1, norm))
        self.count += 1
        self.norm = _add_nonnegative(self.norm, norm)
        budget = _roundoff_budget(operations, norm)
        # The relative rounding model excludes underflow. Allow one
        # smallest representable number for each affected weight product.
        subnormal = np.count_nonzero((absolute < _MIN_NORMAL) & (unweighted != 0.0))
        budget = _add_nonnegative(budget, subnormal * _MIN_SUBNORMAL)
        self.block_budget = _add_nonnegative(self.block_budget, budget)

    def budget(self) -> float:
        # A sequential count also covers additions across causal leaves and
        # blocks evaluated with the compensated summation path.
        reduction = _roundoff_budget(max(0, self.count - 1), self.norm)
        return _add_nonnegative(self.block_budget, reduction)


class _GLRefinement(NamedTuple):
    fine: float
    middle: float
    fine_roundoff: float
    middle_roundoff: float
