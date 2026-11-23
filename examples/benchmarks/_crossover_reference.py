"""Sparse Markov hierarchy for the one-site temporal-kernel benchmarks."""

from __future__ import annotations

from math import prod, sqrt

import numpy as np
from scipy.special import roots_hermite

from validation.hierarchy import Embedding, check_terminates, im


def _kernel_block(kernel, amplitude, width, modes):
    """Return the latent drift, stationary covariance, loading and diffusion."""
    if kernel == "white":
        empty = np.empty((0, 0))
        return empty, empty, np.empty(0), np.empty(0)
    if kernel == "ou":
        rate = 1.0 / width
        return (
            np.array([[-rate]]),
            np.array([[amplitude]]),
            np.array([1.0]),
            np.array([2 * rate * amplitude]),
        )
    if kernel == "matern32":
        rate = sqrt(3.0) / width
        return (
            rate * np.array([[-1.0, 1.0], [0.0, -1.0]]),
            amplitude * np.array([[1.0, 1.0], [1.0, 2.0]]),
            np.array([1.0, 0.0]),
            np.array([0.0, 4 * rate * amplitude]),
        )
    if kernel != "gaussian":
        raise ValueError(f"unknown temporal kernel {kernel!r}")
    if modes is None or modes < 2 or modes % 2:
        raise ValueError("Gaussian embedding needs a positive even mode count")

    nodes, weights = roots_hermite(modes)
    frequencies = sqrt(2.0) * nodes[modes // 2 :] / width
    # A frequency pair contributes twice the weight of its positive node.
    # Both coordinates of its real oscillator have this stationary variance.
    variances = 2 * amplitude * weights[modes // 2 :] / sqrt(np.pi)
    drift = np.zeros((modes, modes))
    even = np.arange(0, modes, 2)
    drift[even, even + 1] = -frequencies
    drift[even + 1, even] = frequencies
    loading = np.zeros(modes)
    loading[even] = 1.0
    return drift, np.diag(np.repeat(variances, 2)), loading, np.zeros(modes)


class _OneSiteHierarchy(Embedding):
    """Use the existing hierarchy query engine with a benchmark-local SDE."""

    def __init__(self, case, modes):
        dimension = case["dimension"]
        gamma = case["gamma"]
        if len(gamma) != dimension:
            raise ValueError("the decay rates must match the physical dimension")
        kernel = case["kernel"]
        amplitude = case["amplitude"]
        block, covariance, loading, diffusion = _kernel_block(
            kernel, amplitude, case["sigma"], modes
        )
        self.hidden_states = len(loading)
        self.physical_dimension = dimension
        self.D = dimension * (1 + self.hidden_states)
        self.n_tags = 1
        self._covariance = covariance
        self._cache = {}
        self._moments = {}
        self._initial_moments = {}
        self._tag = (case["k"],)
        self._legs = tuple(
            (component, time - case["t_min"])
            for component, time in zip(case["components"], case["times"], strict=True)
        )
        if len(self._legs) != case["n"]:
            raise ValueError("the observable legs must match its cumulant order")
        if any(
            component < 0 or component >= dimension or time < 0
            for component, time in self._legs
        ):
            raise ValueError("observable component or time is out of range")

        sde = im.PolySDE(D=self.D, n_tags=self.n_tags)
        zero = self.zero_tag()
        const = im.unit(self.D)
        latent_entries = tuple(zip(*np.nonzero(block)))
        for a, rate in enumerate(gamma):
            sde.drift.append((a, -rate, im.unit(self.D, a), zero))
            if kernel == "white":
                sde.diffusion.append((a, a, amplitude, const, zero))
                continue
            offset = dimension + a * self.hidden_states
            for j in np.flatnonzero(loading):
                sde.drift.append(
                    (a, float(loading[j]), im.unit(self.D, offset + j), zero)
                )
            for i, j in latent_entries:
                sde.drift.append(
                    (offset + i, float(block[i, j]), im.unit(self.D, offset + j), zero)
                )
            for j in np.flatnonzero(diffusion):
                state = offset + j
                sde.diffusion.append((state, state, float(diffusion[j]), const, zero))
        for target, left, right, coefficient in case["drift"]:
            if coefficient:
                sde.drift.append(
                    (target, coefficient, im.unit(self.D, left, right), (1,))
                )
        self._parts = [sde]
        check_terminates(self)

    def _gaussian_moment(self, powers):
        """Isserlis recursion for one independent latent block."""
        if not any(powers):
            return 1.0
        if sum(powers) % 2:
            return 0.0
        if powers in self._initial_moments:
            return self._initial_moments[powers]
        first = next(i for i, count in enumerate(powers) if count)
        remaining = list(powers)
        remaining[first] -= 1
        value = 0.0
        for j, multiplicity in enumerate(remaining):
            covariance = self._covariance[first, j]
            if multiplicity and covariance:
                remaining[j] -= 1
                value += (
                    multiplicity * covariance * self._gaussian_moment(tuple(remaining))
                )
                remaining[j] += 1
        self._initial_moments[powers] = value
        return value

    def initial(self, alpha, tag):
        if any(tag) or any(alpha[: self.physical_dimension]):
            return 0.0
        if not self.hidden_states:
            return 1.0
        step = self.hidden_states
        return prod(
            self._gaussian_moment(alpha[start : start + step])
            for start in range(self.physical_dimension, self.D, step)
        )

    def raw_moment(self, legs, tags):
        key = (tuple(sorted(legs)), tuple(tags))
        if key not in self._moments:
            self._moments[key] = super().raw_moment(*key)
        return self._moments[key]

    def _batch_equal_time(self, time):
        """Solve every distinct subset moment and lower tag together."""
        tags = tuple((order,) for order in range(self._tag[0] + 1))
        subsets = {
            tuple(
                sorted(self._legs[j] for j in range(len(self._legs)) if mask >> j & 1)
            )
            for mask in range(1, 1 << len(self._legs))
        }
        monomials = {
            legs: im.monomial_of(self.D, (component for component, _ in legs))
            for legs in subsets
        }
        targets = [
            (monomial, tag)
            for monomial in sorted(set(monomials.values()))
            for tag in tags
            if (sum(monomial) + tag[0]) % 2 == 0
        ]
        # Changing the sign of the Gaussian noise and quadratic coupling
        # changes the sign of every physical field. Odd-degree/tag pairs
        # therefore vanish exactly.
        values = (
            im.solve(self, targets, [time], initial=self.initial) if targets else {}
        )
        for legs, monomial in monomials.items():
            self._moments[legs, tags] = {
                tag: float(values[monomial, tag][0])
                if (monomial, tag) in values
                else 0.0
                for tag in tags
            }

    def evaluate(self) -> tuple[float, float]:
        times = {time for _, time in self._legs}
        if len(times) == 1:
            self._batch_equal_time(next(iter(times)))
        moments, scale = self.joint_cumulant_with_scale(self._legs, (self._tag,))
        return moments[self._tag], scale


def prepare(case: dict, modes: int | None) -> _OneSiteHierarchy:
    """Prepare a hierarchy evaluator for the coefficient of F at tag `(k,)`."""
    return _OneSiteHierarchy(case, modes)
