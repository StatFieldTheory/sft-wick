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


def contract_pair(
    op1: FieldOperator,
    op2: FieldOperator,
    ito: bool = True,
) -> Optional[Propagator]:
    r"""Contract two field operators and return the propagator, or None if it vanishes.

    Convention for R: physical field index/position is always on the left.
    R_{ij}(x, x') = <phi_i(x) psi_j(x')>_{S_0}

    Args:
        op1: First field operator.
        op2: Second field operator.
        ito: If ``True``, the Itô prescription :math:`\Theta(0)=0` is
            applied: the response propagator vanishes at equal spatial
            points, i.e. :math:`R(x,x)=0`.  This eliminates diagrams
            with equal-point response contractions (e.g. intra-vertex
            tadpoles in local vertices).

    Returns:
        A :class:`~sft_wick.expressions.Propagator`, or ``None`` if the
        contraction vanishes.
    """
    t1, t2 = op1.field_type, op2.field_type

    if t1 == FieldType.RESPONSE and t2 == FieldType.RESPONSE:
        return None

    if t1 == FieldType.PHYSICAL and t2 == FieldType.PHYSICAL:
        return Propagator(
            kind="C",
            index_left=op1.component_index,
            index_right=op2.component_index,
            spatial_left=op1.spatial_arg,
            spatial_right=op2.spatial_arg,
        )

    if t1 == FieldType.PHYSICAL and t2 == FieldType.RESPONSE:
        if ito and op1.spatial_arg == op2.spatial_arg:
            return None
        return Propagator(
            kind="R",
            index_left=op1.component_index,
            index_right=op2.component_index,
            spatial_left=op1.spatial_arg,
            spatial_right=op2.spatial_arg,
        )

    # t1 == RESPONSE, t2 == PHYSICAL: canonical order puts phi first
    if ito and op1.spatial_arg == op2.spatial_arg:
        return None
    return Propagator(
        kind="R",
        index_left=op2.component_index,
        index_right=op1.component_index,
        spatial_left=op2.spatial_arg,
        spatial_right=op1.spatial_arg,
    )
