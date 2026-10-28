"""Build-time analysis of a coupling sum for the ``"factorized"`` backend.

:func:`lower_coupling` lowers the coupling expression of a diagram into
monomials of static coupling tensors, rationals and factors ``i``;
:func:`prefactor` folds the rational prefactor with the response phase;
:func:`component` keeps the real or imaginary part a tensor uses.
:func:`coefficient_pattern` decides rule (4) of
:mod:`sft_wick._contraction_admission` (no two monomials of opposite signs
at any assignment) and counts the nonzero coefficients for the cost model.
:mod:`sft_wick._factorized_coupling` builds its snapshot from these.
"""

from __future__ import annotations

import math

import numpy as np

from ._contraction_admission import (
    _native_dtype,
    _ordinary_label,
    finite_range,
    quarter_turn,
)
from ._dense_contraction import execute, plan_contraction
from .expressions import _NATIVE_NUMERIC_EXPRESSION_CLASSES

# Definition-time class bindings, as the ordered numerical evaluators use.
(Product, Sum, Symbol, Rational, ImaginaryUnit, _) = _NATIVE_NUMERIC_EXPRESSION_CLASSES

#: Assignments evaluated together by the sign check.
_SIGN_CHUNK = 2**20
#: Largest coefficient whose 0/1 zero pattern is kept for the cost model.
_MASK_ELEMENTS = 2**16
#: Lowering limits: expression depth and factors per monomial.
_MAX_DEPTH = 32
_MAX_FACTORS = 24


# ------------------------------------------------------------- lowering


def lower_coupling(expression, static_cv, names, fixed):
    """Lower a coupling sum into its monomials.

    Returns ``(groups, used)``: each group lists one monomial's factors,
    ``("r", value)``, ``("i",)`` or ``("s", slot, slices, labels)``, and
    ``used`` maps coupling names to slots.  ``None`` for any unsupported node.
    """
    monomials = _monomial_nodes(expression)
    if monomials is None:
        return None
    used = {}
    context = (static_cv, names, fixed, used)
    groups = []
    for monomial in monomials:
        factors = []
        if not _lower_node(monomial, factors, 1, context) or len(factors) > _MAX_FACTORS:
            return None
        groups.append(tuple(factors))
    return groups, used


def _monomial_nodes(expression):
    """The terms of a ``Sum`` of ``Product`` nodes, or the expression alone."""
    if type(expression) is not Sum:
        return (expression,)
    if type(expression.terms) is not tuple or not expression.terms:
        return None
    if any(type(monomial) is not Product for monomial in expression.terms):
        return None
    return expression.terms


def _lower_node(node, factors, depth, context):
    """Append the factors of ``node`` to ``factors``; ``False`` if unsupported."""
    if depth > _MAX_DEPTH:
        return False
    kind = type(node)
    if kind is Rational:
        return _lower_rational(node, factors)
    if kind is ImaginaryUnit:
        factors.append(("i",))
        return True
    if kind is Symbol:
        return _lower_symbol(node, factors, *context)
    if kind is not Product or type(node.factors) is not tuple:
        return False
    return all(_lower_node(child, factors, depth + 1, context) for child in node.factors)


def _lower_rational(node, factors):
    if (
        type(node.numerator) is not int
        or type(node.denominator) is not int
        or not node.denominator
    ):
        return False
    try:
        value = node.numerator / node.denominator
    except (OverflowError, ZeroDivisionError):
        return False
    if not math.isfinite(value):
        return False
    factors.append(("r", value))
    return True


def _lower_symbol(node, factors, static_cv, names, fixed, used):
    """A coupling tensor: summation labels become axes, fixed ones slices."""
    if (
        type(node.name) is not str
        or type(node.indices) is not tuple
        or node.name not in static_cv
        or any(not _ordinary_label(label) for label in node.indices)
    ):
        return False
    array = static_cv[node.name]
    if array.ndim != len(node.indices):
        return False
    slices = []
    labels = []
    for label, dim in zip(node.indices, array.shape):
        if label in names:
            axis, size = names[label]
            if dim != size:
                return False
            slices.append(slice(None))
            labels.append(axis)
        elif label in fixed and 0 <= fixed[label] < dim:
            slices.append(fixed[label])
        else:
            return False
    slot = used.setdefault(node.name, len(used))
    factors.append(("s", slot, tuple(slices), tuple(labels)))
    return True


def prefactor(term):
    """``(-i)**n_response`` times the rational prefactor, or ``None``."""
    rational = term.rational_prefactor
    n_response = term.n_response
    e_psi = term.n_external_response
    if (
        type(rational) is not Rational
        or type(rational.numerator) is not int
        or type(rational.denominator) is not int
        or not rational.denominator
        or type(n_response) is not int
        or not 0 <= n_response <= 1024
        or not quarter_turn(e_psi)
    ):
        return None
    try:
        value = [1.0, -1j, -1.0, 1j][n_response % 4] * (
            rational.numerator / rational.denominator
        )
    except (OverflowError, ZeroDivisionError):
        return None
    if not math.isfinite(abs(value)) or value == 0:
        return None
    return value


def component(array):
    """``(component, phase, range)`` of a real or imaginary array."""
    if _native_dtype(array, np.float64):
        part, phase = array, 1 + 0j
    elif not np.any(array.real):
        part, phase = array.imag, 1j
    elif not np.any(array.imag):
        part, phase = array.real, 1 + 0j
    else:
        return None
    scanned = finite_range(part)
    if scanned is None:
        return None
    return part, phase, scanned


# ------------------------------------------------- sign agreement, rule (4)


def coefficient_pattern(monomials, components, shape, available, max_work):
    """Sign agreement and zero pattern of the summed coefficient.

    The sign of monomial ``t`` at assignment ``a`` is the sign of its real
    phase times the signs of its factor entries, computed exactly on int8
    sign arrays.  Returns ``(nonzero, mask)``: the number of assignments
    where some monomial is nonzero and, for at most ``_MASK_ELEMENTS``
    assignments, that pattern as a 0/1 float64 array.  When this
    per-assignment check would exceed ``max_work`` element operations or
    ``available`` bytes, agreement must follow from the factor signs alone
    (:func:`_signs_agree`); ``nonzero`` is then the largest count of one
    monomial, a lower bound that is exact for a single monomial, and
    ``mask`` is ``None``.  Returns ``None`` when two monomials have opposite
    signs at some assignment, or when neither check proves agreement.
    """
    assignments = math.prod(shape)
    work = assignments * sum(len(m.factors) + 1 for m in monomials)
    keep_mask = assignments <= _MASK_ELEMENTS
    scan_bytes = available - sum(part.size for part, _, _ in components)
    scan_bytes -= 8 * assignments if keep_mask else 0
    chunked = _chunks(shape, min(_SIGN_CHUNK, scan_bytes // 4)) if scan_bytes >= 4 else None
    if work > max_work or chunked is None:
        if len(monomials) > 1 and not _signs_agree(monomials, components):
            return None
        nonzero = max(
            _nonzero_count(monomial, components, shape, available) for monomial in monomials
        )
        return nonzero, None
    return _checked_pattern(monomials, components, shape, chunked, keep_mask)


def _chunks(shape, limit):
    """Prefix length ``p`` and the prefix assignments of a chunked scan.

    The trailing ``shape[p:]`` holds at most ``limit`` elements; ``None`` when
    that takes more than 4096 chunks.
    """
    for p in range(len(shape) + 1):
        if math.prod(shape[p:]) <= limit:
            if math.prod(shape[:p]) > 4096:
                return None
            return p, np.ndindex(*shape[:p])
    return None


def _checked_pattern(monomials, components, shape, chunked, keep_mask):
    """The per-assignment check of :func:`coefficient_pattern`, chunk by chunk."""
    p, prefixes = chunked
    signs = [np.sign(part).astype(np.int8) for part, _, _ in components]
    tail = list(range(p + 1, len(shape) + 1))
    ones = np.broadcast_to(np.ones((), dtype=np.int8), shape[p:])
    mask = np.zeros(shape, dtype=np.float64) if keep_mask else None
    nonzero = 0
    for prefix in prefixes:
        present = _chunk_signs(monomials, signs, prefix, p, tail, ones)
        if present is None:
            return None
        nonzero += int(np.count_nonzero(present))
        if mask is not None:
            mask[prefix] = present
    if mask is not None:
        mask.flags.writeable = False
    return nonzero, mask


def _chunk_signs(monomials, signs, prefix, p, tail, ones):
    """Assignments of one chunk where some monomial is nonzero, or ``None``
    when two monomials have opposite signs at one of them."""
    shape = ones.shape
    positive = np.zeros(shape, dtype=bool)
    negative = np.zeros(shape, dtype=bool)
    for monomial in monomials:
        if monomial.phase == 0.0:
            continue
        operands = [ones, tail]
        for slot, _, slices, labels in monomial.factors:
            index = tuple(
                prefix[label - 1] if label <= p else slice(None) for label in labels
            )
            kept = [label for label in labels if label > p]
            operands += [signs[slot][slices][index], kept]
        value = np.einsum(*operands, tail)
        if monomial.phase < 0.0:
            positive |= value < 0
            negative |= value > 0
        else:
            positive |= value > 0
            negative |= value < 0
        # Stop at the first monomial that conflicts with an earlier one.
        if len(monomials) > 1 and np.any(positive & negative):
            return None
    return positive | negative


def _factor_sign(part):
    """1 or -1 when every nonzero entry has that sign, 0 when all are zero,
    ``None`` when the entries hold both signs (entries are finite)."""
    if part.size == 0:
        return 0
    high = float(np.max(part))
    low = float(np.min(part))
    if low >= 0.0:
        return 1 if high > 0.0 else 0
    if high <= 0.0:
        return -1
    return None


def _signs_agree(monomials, components):
    """Whether the factor signs alone prove that no two monomials differ in sign.

    A monomial whose factors each hold one sign (zeros allowed) is, at every
    assignment, zero or of one fixed sign: its phase's sign times its factor
    signs.  Returns ``True`` when every monomial that is not identically zero
    has the same fixed sign.  The work is one pass over each distinct factor.
    """
    signs = {}
    common = 0
    for monomial in monomials:
        if monomial.phase == 0.0:
            continue
        sign = 1 if monomial.phase > 0.0 else -1
        for slot, _, slices, _ in monomial.factors:
            # Slices are hashable only from Python 3.12.
            key = (slot, tuple(None if type(item) is slice else item for item in slices))
            if key not in signs:
                signs[key] = _factor_sign(components[slot][0][slices])
            if signs[key] is None:
                return False
            sign *= signs[key]
        if sign and common and sign != common:
            return False
        common = common or sign
    return True


def _nonzero_count(monomial, components, shape, available):
    """Assignments at which every factor of ``monomial`` is nonzero.

    Contracts the 0/1 patterns of the factors along a planned path, so the
    work is that of the contraction, not of the assignment tensor.  Returns
    0, a valid lower bound, when the patterns and the path do not fit
    ``available`` bytes.
    """
    if monomial.phase == 0.0:
        return 0
    operands = []
    used = set()
    for slot, _, slices, labels in monomial.factors:
        operands.append(((components[slot][0][slices] != 0).astype(np.float64), labels))
        used.update(labels)
    free = math.prod(dim for axis, dim in enumerate(shape, 1) if axis not in used)
    if not operands:
        return free
    # A one-element sample axis gives the planner its required output label.
    operands.append((np.ones(1), (0,)))
    plan = plan_contraction(
        tuple(array.shape for array, _ in operands),
        tuple(labels for _, labels in operands),
    )
    if plan is None:
        return 0
    steps, _, workspace = plan
    if 8 * (workspace + sum(array.size for array, _ in operands)) > available:
        return 0
    return free * int(execute(operands, steps)[0])
