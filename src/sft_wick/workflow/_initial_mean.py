"""Translate local polynomial interactions about the free initial mean."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class _PropagatedMean:
    response: Callable
    mean: np.ndarray

    def __call__(self, times):
        return self.response(times) @ self.mean


@dataclass(frozen=True)
class _ObservableMean:
    propagated: _PropagatedMean
    vectorized = True

    def __call__(self, positions, times):
        # The external factor has one absorbed R slot. Cancel its response
        # phase (-i), without adding a new perturbative insertion.
        values = 1j * self.propagated(times[0])
        return values[..., 0] if self.propagated.mean.size == 1 else values


@dataclass(frozen=True)
class _TranslatedVertex:
    name: str
    original: str
    n_psi: int
    n_legs: int
    retained: int

    def fields(self, phi, psi):
        return (psi,) * self.n_psi + (phi,) * self.retained


@dataclass(frozen=True)
class _TranslatedCoupling:
    vertex: _TranslatedVertex
    bare_msr: np.ndarray | Callable
    propagated: _PropagatedMean
    vectorized = True

    def __call__(self, positions, times):
        mean = self.propagated(times[0])
        n_samples, n_components = mean.shape
        vertex = self.vertex
        coefficient = self.bare_msr
        dynamic = callable(coefficient)
        if dynamic:
            if getattr(coefficient, "vectorized", False):
                coefficient = np.asarray(coefficient(positions, times))
            else:
                coefficient = np.stack([
                    np.array(coefficient(positions[:, i], times[:, i]), copy=True)
                    for i in range(n_samples)
                ])
        shape = ((n_samples,) if dynamic else ()) + (n_components,) * vertex.n_legs
        if n_components == 1:
            # Scalar L0 coupling symbols have no component indices. Every
            # choice of retained legs gives the same scalar monomial.
            coefficient = np.asarray(coefficient).reshape(shape[:int(dynamic)])
            degree = vertex.n_legs - vertex.n_psi
            return comb(degree, vertex.retained) * coefficient * (
                mean[:, 0] ** (degree - vertex.retained)
            )
        if np.shape(coefficient) != shape:
            raise ValueError(
                f"vertex {vertex.original!r} returned coupling shape "
                f"{np.shape(coefficient)}, expected {shape}"
            )
        axes = list(range(1, vertex.n_legs + 1))
        response_axes = axes[:vertex.n_psi]
        physical_axes = axes[vertex.n_psi:]
        value = np.zeros(
            (n_samples,) + (n_components,) * (vertex.n_psi + vertex.retained),
            dtype=np.result_type(coefficient, mean),
        )
        for kept in combinations(physical_axes, vertex.retained):
            operands = [coefficient, [0, *axes] if dynamic else axes]
            for index in physical_axes:
                if index not in kept:
                    operands.extend((mean, [0, index]))
            value += np.einsum(*operands, [0, *response_axes, *kept], optimize=True)
        return value


def translated_vertex_names(shapes, taken):
    """Return a unique symbol for the observable mean and one per lower rank."""
    taken = set(taken)

    def reserve(name):
        while name in taken:
            name += "_"
        taken.add(name)
        return name

    mean = reserve("initial_mean")
    vertices = tuple(
        _TranslatedVertex(reserve(f"{name}_mean_{k}"), name, n_psi, n_legs, k)
        for name, n_psi, n_legs in shapes
        for k in range(n_legs - n_psi)
    )
    return mean, vertices
