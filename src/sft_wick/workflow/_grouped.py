"""Ordered mixed vertices at shared points or times."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .._contraction_admission import DEFAULT_CONTRACTION_MAX_BYTES
from ..vertices import Vertex
from .specs import (
    _flag, _msr_factor, _nested_tuple, _real_vertex_tensor, _shape_problem,
    _VertexTensorCoupling,
)


@dataclass(frozen=True, eq=False)
class GroupedVertex:
    """A bare interaction tensor with one axis for each ordered field slot.

    Each entry of ``fields`` is ``"psi"`` (response) or ``"phi"`` (physical).
    In ``group_mode="spacetime"``, slots with the same group ID share a
    position and a time. The callable takes ``(positions, times)`` with
    one entry per group, in ascending group ID. With ``group_mode="time"``
    it takes one entry per field slot; grouped slots have equal times and
    can have different positions. Independent groups integrate independently.
    A lag ``(d, a, tau)`` fixes ``t[d] = t[a] - tau``.

    A coupling has shape ``(N,) * len(fields)``. A vectorized callable
    accepts coordinate arrays with samples on the last axis and returns a
    leading sample axis. The spec multiplies a bare tensor by −iᵐ/m! for
    m response slots.
    """

    name: str
    fields: tuple[str, ...]
    coupling: Any
    leg_groups: tuple[int, ...]
    group_mode: str = "spacetime"
    group_lags: tuple[tuple[int, int, float], ...] = ()
    coupling_vectorized: bool = False

    def __post_init__(self):
        from ..vertices import normalize_leg_groups

        if not isinstance(self.name, str) or not self.name:
            raise ValueError("GroupedVertex requires a non-empty name.")
        if isinstance(self.fields, str):
            raise ValueError("fields must be a sequence of 'phi' and 'psi' slots.")
        fields = tuple(self.fields)
        if any(not isinstance(f, str) for f in fields) or not (
            "psi" in fields and set(fields) <= {"phi", "psi"}
        ):
            raise ValueError("fields must contain only 'phi' and 'psi', with a response slot.")
        object.__setattr__(self, "fields", fields)
        groups, lags = normalize_leg_groups(
            len(fields), self.leg_groups, self.group_mode, self.group_lags
        )
        if not groups:
            raise ValueError("GroupedVertex requires a group ID for every field slot.")
        object.__setattr__(self, "leg_groups", groups)
        object.__setattr__(self, "group_lags", lags)
        vectorized = _flag(self.coupling_vectorized, "coupling_vectorized", "GroupedVertex")
        object.__setattr__(self, "coupling_vectorized", vectorized)
        if self.n_legs > 64 - int(vectorized):
            raise ValueError("GroupedVertex has too many axes for a NumPy tensor.")
        if not callable(self.coupling):
            if vectorized:
                raise ValueError("coupling_vectorized requires a callable coupling.")
            raw = _real_vertex_tensor(self.coupling, self.name)
            if raw.ndim == 0:
                raw = raw.reshape((1,) * self.n_legs)
            if raw.ndim != self.n_legs:
                raise ValueError("GroupedVertex coupling needs one axis for each field slot.")
            object.__setattr__(self, "coupling", _nested_tuple(raw))

    @property
    def n_legs(self) -> int:
        return len(self.fields)

    @property
    def msr_factor(self) -> complex:
        return _msr_factor(self.fields.count("psi"))

    def shape_problem(self, n_components: int) -> str | None:
        if isinstance(self.coupling, _RateCovariance):
            n = self.coupling.tensor.shape[0]
            if n != n_components:
                return f"the rate matrix has {n} components and the field has {n_components}"
        if callable(self.coupling):
            return None
        return _shape_problem(np.shape(self.coupling), (n_components,) * self.n_legs, n_components)

    def build_vertex(self, phi, psi) -> Vertex:
        return Vertex(
            tuple(psi if slot == "psi" else phi for slot in self.fields),
            self.name, local=False, leg_groups=self.leg_groups,
            group_mode=self.group_mode, group_lags=self.group_lags,
        )

    def build_msr_coupling(self, n_components: int):
        if callable(self.coupling):
            return _VertexTensorCoupling(self, n_components)
        return self.msr_factor * np.asarray(self.coupling)


@dataclass(frozen=True, eq=False)
class _RateCovariance:
    kernel: Callable
    tensor: np.ndarray
    vectorized: bool

    # Many finite-time covariances have a cusp at coincident times (e.g. OU).
    has_coincident_time_kinks = True

    def __call__(self, positions, times):
        value = _real_vertex_tensor(self.kernel(positions, times), "coloured covariance")
        expected = (np.shape(times)[-1],) if self.vectorized else ()
        if value.shape != expected:
            raise ValueError(f"coloured covariance must return shape {expected}, got {value.shape}")
        return value[..., None, None, None, None] * self.tensor


def ColouredMultiplicativeNoise(
    kernel: Callable, g: Any = 1.0, *, name: str = "rate",
    coupling_vectorized: bool = False,
) -> GroupedVertex:
    """Average a zero-mean scalar Gaussian rate with covariance ``kernel``.

    For a term ξ(t,x) Gφ in the equation of motion, supply a square
    matrix ``g=G`` (a number for one component) and the covariance
    ``kernel(positions, times)`` of ξ at two points. It returns a scalar
    per sample. One insertion represents two powers of G; the vertex
    has the positive action coefficient ½ K G⊗G.
    """
    if not callable(kernel):
        raise TypeError("kernel must be a covariance callable of (positions, times).")
    vectorized = _flag(coupling_vectorized, "coupling_vectorized", "ColouredMultiplicativeNoise")
    matrix = _real_vertex_tensor(g, "rate matrix")
    if matrix.ndim == 0:
        matrix = matrix.reshape(1, 1)
    if matrix.ndim != 2 or not matrix.size or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("g must be a finite square matrix, or a scalar for one component.")
    if matrix.size ** 2 * matrix.itemsize > DEFAULT_CONTRACTION_MAX_BYTES:
        raise ValueError("The dense fourth-rank rate tensor exceeds the supported workspace.")
    with np.errstate(over="ignore"):
        tensor = np.multiply.outer(matrix, matrix)
    if not np.all(np.isfinite(tensor)):
        raise ValueError("The fourth-rank rate tensor exceeds floating-point range.")
    return GroupedVertex(
        name=name, fields=("psi", "phi", "psi", "phi"), leg_groups=(0, 0, 1, 1),
        coupling=_RateCovariance(kernel, tensor, vectorized),
        coupling_vectorized=vectorized,
    )
