"""Lower a spatially constant initial law to propagated covariance and vertices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .specs import NonLocalVertex
from ._covariance import is_covariance


def _symmetric_tensor(value, order):
    if np.iscomplexobj(value):
        raise ValueError(f"initial tensor of order {order} must be real")
    tensor = np.array(value, dtype=float, copy=True)
    if tensor.ndim != order or not tensor.size or len(set(tensor.shape)) != 1:
        raise ValueError(f"initial tensor of order {order} must have shape (N,)*{order}")
    if not np.isfinite(tensor).all():
        raise ValueError(f"initial tensor of order {order} must be finite")
    if any(
        not np.array_equal(tensor, tensor.swapaxes(axis, axis + 1))
        for axis in range(order - 1)
    ):
        raise ValueError(f"initial tensor of order {order} must be symmetric")
    tensor.flags.writeable = False
    return tensor


@dataclass(frozen=True)
class InitialState:
    """Spatially constant initial state, independent of later noise.

    ``covariance`` is the free initial covariance at ``System.t_min``.
    ``cumulants`` contains tensors of distinct ranks 3 or higher. Each
    higher cumulant defines an interaction named ``I{rank}`` and counts
    once toward perturbative order each time it occurs.
    ``mean`` is the optional initial mean vector. Its free evolution
    supplies the deterministic part of the observable and translates
    the local polynomial interactions.
    """

    covariance: np.ndarray | None = None
    cumulants: tuple = ()
    mean: np.ndarray | None = None

    def __post_init__(self):
        if self.mean is not None:
            object.__setattr__(self, "mean", _symmetric_tensor(self.mean, 1))
        if self.covariance is not None:
            covariance = _symmetric_tensor(self.covariance, 2)
            if not is_covariance(covariance):
                raise ValueError("initial covariance must be positive semidefinite")
            object.__setattr__(self, "covariance", covariance)
        orders = set()
        tensors = []
        for value in self.cumulants:
            order = np.ndim(value)
            if order < 3 or order in orders:
                raise ValueError("initial cumulants must have distinct orders of at least 3")
            orders.add(order)
            tensors.append(_symmetric_tensor(value, order))
        object.__setattr__(self, "cumulants", tuple(tensors))

    def _check_components(self, n_components):
        for tensor in (self.mean, self.covariance, *self.cumulants):
            if tensor is not None and tensor.shape[0] != n_components:
                raise ValueError(
                    f"initial tensor of order {tensor.ndim} has {tensor.shape[0]} "
                    f"components, but the field has {n_components}"
                )

    @property
    def _named_cumulants(self):
        return tuple((f"I{tensor.ndim}", tensor) for tensor in self.cumulants)

    def _vertices(self, response):
        return tuple(
            NonLocalVertex(
                name,
                order=tensor.ndim,
                coupling=_InitialCumulant(response, tensor),
                coupling_vectorized=True,
                already_R_contracted=True,
                leg_symmetry="declared",
            )
            for name, tensor in self._named_cumulants
        )


@dataclass(frozen=True)
class _InitialResponse:
    r_time: Callable
    t_min: float
    iso_r: bool
    n_components: int

    def __call__(self, times):
        times = np.asarray(times, dtype=float)
        times_flat = times.reshape(-1)
        identity = np.eye(self.n_components)
        out = np.zeros((len(times_flat), self.n_components, self.n_components))
        out[times_flat == self.t_min] = identity
        active = times_flat > self.t_min
        if np.any(active):
            later = times_flat[active]
            if getattr(self.r_time, "vectorized", False):
                values = np.asarray(self.r_time(later, np.full_like(later, self.t_min)))
            else:
                unique, inverse = np.unique(later, return_inverse=True)
                values = np.array([
                    np.array(self.r_time(float(t), self.t_min), copy=True)
                    for t in unique
                ])[inverse]
            expected = (len(later),) if self.iso_r else (len(later), *identity.shape)
            if values.shape != expected:
                raise ValueError(
                    f"R_time returned shape {values.shape} for initial state "
                    f"propagation, expected {expected}"
                )
            out[active] = values[:, None, None] * identity if self.iso_r else values
        return out.reshape(times.shape + identity.shape)


@dataclass(frozen=True)
class _InitialCovariance:
    response: _InitialResponse
    covariance: np.ndarray

    def __call__(self, t1, t2):
        left = self.response(t1)
        right = self.response(t2)
        return left @ self.covariance @ right.swapaxes(-1, -2)


@dataclass(frozen=True)
class _InitialCumulant:
    response: _InitialResponse
    cumulant: np.ndarray

    def __call__(self, positions, times):
        order = self.cumulant.ndim
        if self.response.n_components == 1:
            responses = self.response(times)[..., 0, 0]
            result = self.cumulant.item() * responses.prod(axis=0)
            return result.reshape((len(result),) + (1,) * order)
        internal = list(range(1, order + 1))
        external = list(range(order + 1, 2 * order + 1))
        operands = [self.cumulant, internal]
        for row, i, a in zip(times, internal, external):
            operands.extend((self.response(row), [0, a, i]))
        return np.einsum(*operands, [0, *external], optimize=True)
