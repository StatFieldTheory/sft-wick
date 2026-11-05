"""Propagator contraction rules for MSR field theory.

Defines how pairs of field operators contract:
  <phi_i(x) phi_j(x')>_{S_0} = C_{ij}(x, x')
  <phi_i(x) psi_j(x')>_{S_0} = R_{ij}(x, x')
  <psi_i(x) psi_j(x')>_{S_0} = 0
"""

from __future__ import annotations

from typing import Optional

from .expressions import Propagator
from .fields import FieldOperator, FieldType


def propagator(kind: str, left: FieldOperator, right: FieldOperator) -> Propagator:
    """The propagator of ``kind`` from ``left`` to ``right``.

    The caller has chosen the order: an R is stored with its φ on the left.
    """
    return Propagator(
        kind=kind,
        index_left=left.component_index,
        index_right=right.component_index,
        spatial_left=left.spatial_arg,
        spatial_right=right.spatial_arg,
    )


def contract_pair(
    op1: FieldOperator,
    op2: FieldOperator,
) -> Optional[Propagator]:
    r"""Contract two field operators and return the propagator, or None if it vanishes.

    Convention for R: physical field index/position is always on the left.
    R_{ij}(x, x') = <phi_i(x) psi_j(x')>_{S_0}

    Args:
        op1: First field operator.
        op2: Second field operator.

    Returns:
        A :class:`~sft_wick.expressions.Propagator`, or ``None`` if the
        contraction vanishes: two :math:`\psi` operators, or an R that
        :func:`~sft_wick.wick.r_vanishes` drops, which for one edge is the
        Itô rule :math:`R(x,x)=0`.
    """
    t1, t2 = op1.field_type, op2.field_type

    if t1 == FieldType.RESPONSE and t2 == FieldType.RESPONSE:
        return None

    if t1 == FieldType.PHYSICAL and t2 == FieldType.PHYSICAL:
        return propagator("C", op1, op2)

    from .wick import r_vanishes

    # Canonical order puts phi first
    phi_op, psi_op = (op1, op2) if t1 == FieldType.PHYSICAL else (op2, op1)
    if r_vanishes([(phi_op.spatial_arg, psi_op.spatial_arg)]):
        return None
    return propagator("R", phi_op, psi_op)
