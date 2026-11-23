"""Specs for local polynomial Kramers–Moyal generator terms."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..fields import Field
from ..vertices import Vertex
from ._messages import shown
from .specs import (
    _flag, _msr_factor, _nested_tuple, _real_vertex_tensor, _shape_problem,
    _VertexTensorCoupling as _KMCoupling,  # preserves the legacy pickle name
)


@dataclass(frozen=True)
class KramersMoyalVertex:
    r"""One local polynomial coefficient of an Itô generator.

    A vertex with ``m`` response legs represents the bare coefficient of
    ``D^{(m)}_{a_1 ... a_m}(φ) ∂_{a_1} ... ∂_{a_m} / m!``. This spec gives
    one homogeneous polynomial term in ``D^{(m)}``. Its first ``m`` tensor
    axes are response indices and any remaining axes multiply physical
    fields at the same spacetime point. The tensor carries no factorials
    or MSR phase. The spec applies ``−i^m/m!`` when it lowers the coupling.

    Finite higher-order KM expansions give formal moment coefficients.
    For a jump law, keep every derivative order reached by the moments
    and perturbative orders under study.

    Args:
        name: unique coupling name within the system.
        n_psi: number of response legs, at least one.
        coupling: bare finite real tensor with all axes of length ``N``
            for ``N`` field components, or a callable ``f(x_list, t_list)``
            returning that tensor. Each input holds the vertex's single
            spacetime point.
        rank: total number of legs. Required for a callable and inferred
            from the tensor if omitted.
        coupling_vectorized: a callable can accept inputs with shape
            ``(1, n_samples)`` and return ``(n_samples,) + (N,)*rank``.
    """

    name: str
    n_psi: int
    coupling: Any
    rank: int | None = None
    coupling_vectorized: bool = False

    def __post_init__(self) -> None:
        where = f"KramersMoyalVertex(name={shown(self.name)})"
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"{where}: name must be a non-empty string.")
        if (
            isinstance(self.n_psi, (bool, np.bool_))
            or not isinstance(self.n_psi, (int, np.integer))
            or self.n_psi < 1
        ):
            raise ValueError(f"{where}: n_psi must be a positive integer.")
        object.__setattr__(self, "n_psi", int(self.n_psi))
        vectorized = _flag(self.coupling_vectorized, "coupling_vectorized", where)
        object.__setattr__(self, "coupling_vectorized", vectorized)
        rank = self.rank
        if rank is not None:
            if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, (int, np.integer)):
                raise ValueError(f"{where}: rank must be an integer.")
            rank = int(rank)
            object.__setattr__(self, "rank", rank)

        if callable(self.coupling):
            if rank is None:
                raise ValueError(f"{where}: a callable coupling requires rank.")
        else:
            if vectorized:
                raise ValueError(f"{where}: coupling_vectorized requires a callable coupling.")
            array = _real_vertex_tensor(self.coupling, where)
            if not array.size:
                raise ValueError(f"{where}: coupling tensor axes must be nonempty.")
            if rank is not None and rank != array.ndim:
                raise ValueError(
                    f"{where}: rank={rank} but the coupling tensor has {array.ndim} axes."
                )
            rank = array.ndim
            object.__setattr__(self, "coupling", _nested_tuple(array))

        if rank < self.n_psi:
            raise ValueError(f"{where}: rank must be at least n_psi={self.n_psi}.")
        max_rank = 64 - int(vectorized)
        if rank > max_rank:
            raise ValueError(f"{where}: rank must be at most {max_rank} for a NumPy tensor.")

    @property
    def n_legs(self) -> int:
        """Total number of response and physical legs."""
        return self.rank if self.rank is not None else np.ndim(self.coupling)

    @property
    def msr_factor(self) -> complex:
        """The factor ``−i^n_psi/n_psi!`` for this coupling."""
        return _msr_factor(self.n_psi)

    def shape_problem(self, n_components: int) -> str | None:
        """Why a static coupling does not fit ``n_components``, or ``None``."""
        if callable(self.coupling):
            return None
        shape = np.shape(self.coupling)
        return _shape_problem(shape, (int(n_components),) * self.n_legs, n_components)

    def build_vertex(self, phi: Field, psi: Field) -> Vertex:
        """Lower to a local vertex with response legs first."""
        return Vertex(
            fields=(psi,) * self.n_psi + (phi,) * (self.n_legs - self.n_psi),
            coupling=self.name,
            local=True,
        )

    def build_msr_coupling(self, n_components: int | None = None) -> Any:
        """Apply the MSR factor and wire the callable's component shape check."""
        if callable(self.coupling):
            return _KMCoupling(self, n_components)
        return self.msr_factor * np.asarray(self.coupling)

    @property
    def msr_coupling(self) -> Any:
        """Apply the MSR factor to the coupling or to each callable sample."""
        return self.build_msr_coupling()
