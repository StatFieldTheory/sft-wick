"""The ``"einsum"`` backend and the contraction planning both backends share.

``"einsum"`` contracts the prepared coefficient array of a diagram with its
propagator batches (:func:`try_static_contraction`).  Both opt-in backends
plan their pairwise contractions here (:func:`plan_call`,
:func:`plan_groups`, :func:`plan_contraction`) and run them through
:func:`finish`.  Their accuracy contract and admission rules are stated in
:mod:`sft_wick._contraction_admission`, and the kernel layout is built by
:mod:`sft_wick._contraction_layout`.

Both backends apply the cost rule through this module's
``backend_is_cheaper``, so replacing that name here replaces the rule of
both; the test suite does so to run fixtures the rule declines.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import math
from typing import NamedTuple

import numpy as np

# DEFAULT_CONTRACTION_MAX_BYTES and validate_contraction_options are
# imported from this module by evaluate, perturbation and the workflow.
from ._contraction_admission import (  # noqa: F401
    DEFAULT_CONTRACTION_MAX_BYTES,
    backend_is_cheaper,
    call_inputs_ok,
    coefficient_cost,
    count_cost,
    coupling_array_ok,
    deferred_bytes,
    deferred_cost,
    einsum_buffers,
    einsum_workspace,
    finish_workspace,
    finite_range,
    kernel_cost,
    label_axes,
    minimum_backend_cost,
    ordered_cost,
    products_admit,
    rotated_real_coefficient,
    step_bytes,
    step_cost,
    validate_contraction_options,
)
from ._contraction_layout import (
    KernelLayout,
    apply_classes,
    deferred_values,
    kernel_layout,
    label_classes,
    restricted_supports,
    sample_last,
    supports_hold,
)

#: Operands of one planned product; the pairwise search costs O(k**3).
_MAX_OPERANDS = 64


# ------------------------------------------------------------ planning


def _label_sizes(shapes, labels):
    """``{label: size}`` over all operands, or ``None`` on a rank or size clash."""
    dims = {}
    for shape, indices in zip(shapes, labels):
        if len(shape) != len(indices):
            return None
        for label, size in zip(indices, shape):
            if dims.setdefault(label, size) != size:
                return None
    return dims


def _best_pair(sets, counts, dims):
    """``((disjoint, loop, out, i, j), kept)`` of the next pairwise step.

    The pair with a common label, then the smallest loop space, then the
    smallest result; ``kept`` holds the sample label and every label another
    operand still carries.
    """
    best = None
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            involved = sets[i] | sets[j]
            kept = frozenset(
                label
                for label in involved
                if label == 0
                or counts[label] > (label in sets[i]) + (label in sets[j])
            )
            key = (
                sets[i].isdisjoint(sets[j]),
                math.prod(dims[label] for label in involved),
                math.prod(dims[label] for label in kept),
                i,
                j,
            )
            if best is None or key < best[0]:
                best = (key, kept)
    return best


@lru_cache(maxsize=4096)
def plan_contraction(shapes, labels):
    """Pairwise contraction path of one product sum, from shapes only.

    At each step two operands that share a label are contracted (two
    operands without a common label only when no other pair is left): the
    pair with the smallest loop space, then the smallest result; labels no
    other operand carries are summed.  Restricting the search to pairs with
    a common label keeps the sample-free coupling tensors of a factorized
    monomial from forming outer products with each other.  Returns
    ``(steps, flops, workspace)`` or ``None``: ``steps`` holds ``(i, j,
    kept)`` with the positions of the two operands in the current list (the
    result is appended) and the result labels, ordered with the sample label
    ``0`` last; ``flops`` counts two operations per point of every step's
    loop space; ``workspace`` counts every step's result elements.  The final
    operand must carry only the sample label.
    """
    if len(shapes) > _MAX_OPERANDS:
        return None
    dims = _label_sizes(shapes, labels)
    if dims is None:
        return None
    sets = [frozenset(indices) for indices in labels]
    counts = {}
    for indices in sets:
        for label in indices:
            counts[label] = counts.get(label, 0) + 1
    steps = []
    flops = workspace = 0
    while len(sets) > 1:
        (_, loop, out, i, j), kept = _best_pair(sets, counts, dims)
        for label in sets[i] | sets[j]:
            counts[label] -= (label in sets[i]) + (label in sets[j])
        for label in kept:
            counts[label] += 1
        order = tuple(sorted(kept - {0})) + ((0,) if 0 in kept else ())
        steps.append((i, j, order))
        flops += 2 * loop
        workspace += out
        sets = [sets[k] for k in range(len(sets)) if k not in (i, j)] + [kept]
    if sets != [frozenset({0})]:
        return None
    return tuple(steps), flops, workspace


@lru_cache(maxsize=4096)
def planned_path(shapes, labels):
    """:func:`plan_contraction` and the points of its largest step's loop space.

    The loop space of a step spans every label of its two operands; NumPy's
    iterator buffers at most that many entries per operand
    (:func:`~sft_wick._contraction_admission.einsum_buffers`).  Returns
    ``(None, 0)`` when planning fails.
    """
    plan = plan_contraction(shapes, labels)
    if plan is None:
        return None, 0
    dims = {}
    for shape, indices in zip(shapes, labels):
        dims.update(zip(indices, shape))
    sets = [frozenset(indices) for indices in labels]
    largest = 0
    for i, j, kept in plan[0]:
        involved = sets[i] | sets[j]
        largest = max(largest, math.prod(dims[label] for label in involved))
        sets = [sets[k] for k in range(len(sets)) if k not in (i, j)] + [frozenset(kept)]
    return plan, largest


def execute(operands, steps):
    """Run a planned pairwise contraction; returns the sample vector."""
    operands = list(operands)
    for i, j, kept in steps:
        (left, left_labels), (right, right_labels) = operands[i], operands[j]
        operands = [operands[k] for k in range(len(operands)) if k not in (i, j)]
        operands.append(
            (np.einsum(left, list(left_labels), right, list(right_labels), list(kept)), kept)
        )
    ((result, _),) = operands
    return result


# ------------------------------------------------- live-assignment count


def _count_operands(dims, mask_shape, restricted):
    """Shapes and labels of the live-assignment count, from shapes alone.

    The operands are the restricted supports, the coefficient mask (when
    ``mask_shape`` is not ``None``) and a one-element sample axis, which
    gives the planner its required output label.  Only called with at least
    one restricted support.
    """
    shapes = [view.shape for view, _ in restricted]
    labels = [indices for _, indices in restricted]
    if mask_shape is not None:
        shapes.append(mask_shape)
        labels.append(tuple(dims))
    shapes.append((1,))
    labels.append((0,))
    return tuple(shapes), tuple(labels)


def count_bytes(dims, mask_shape, restricted):
    """Bytes the live-assignment count allocates, or ``None`` if it cannot plan.

    Float64 copies of the restricted supports, the one-element sample axis,
    the result of every planned step and the iterator buffers of its
    largest step; nothing when no support excludes an assignment (the count
    is then a product of sizes or a nonzero count of the mask).
    """
    if not restricted:
        return 0
    plan, loop = planned_path(*_count_operands(dims, mask_shape, restricted))
    if plan is None:
        return None
    copies = sum(view.size for view, _ in restricted) + 1
    return 8 * (copies + plan[2]) + einsum_buffers(loop)


def live_assignments(dims, coefficient_mask, supports, *, restricted=None):
    """Number of assignments the ordered fold multiplies out, and its cost.

    ``dims`` maps the summation labels to sizes, ``coefficient_mask`` is a
    0/1 float array over all of them in order (or ``None`` for a dense
    coefficient) and ``supports`` the resolved R supports (or
    ``restricted``, their :func:`restricted_supports`).  The count is
    exact (float64 holds integers below 2**53).  It is contracted along a
    planned pairwise path, so disjoint supports cost the sum, not the
    product, of their sizes.  The arrays it allocates take
    :func:`count_bytes`.  Returns ``(count, estimated cost)``.
    """
    if restricted is None:
        restricted = restricted_supports(supports)
    used = set().union(*(indices for _, indices in restricted))
    if coefficient_mask is not None:
        if not restricted:
            return int(np.count_nonzero(coefficient_mask)), 0.0
        used.update(dims)
    free = math.prod(size for label, size in dims.items() if label not in used)
    if not restricted:
        return free, 0.0
    mask_shape = None if coefficient_mask is None else coefficient_mask.shape
    plan = plan_contraction(*_count_operands(dims, mask_shape, restricted))
    if plan is None:
        return None, math.inf
    operands = [(view.astype(np.float64), indices) for view, indices in restricted]
    if coefficient_mask is not None:
        operands.append((coefficient_mask, tuple(dims)))
    operands.append((np.ones(1), (0,)))
    steps, flops, _ = plan
    count = execute(operands, steps)
    return free * int(count[0]), count_cost(len(steps), flops)


# ------------------------------------------------------ group planning


@dataclass(frozen=True)
class Contraction:
    """A planned contraction of one or more coefficient groups.

    ``workspace`` is the largest number of bytes the arrays allocated by
    :func:`~sft_wick._contraction_layout.kernel_layout` and :func:`finish`
    hold at one time, not counting arrays the caller holds while
    :func:`finish` runs.
    """

    groups: tuple  # ((phase, ((array, labels), ...), steps), ...)
    # ((array or deferred position, sample-last labels, read or None), ...)
    kernels: tuple
    cost: float
    workspace: int


class _KernelPlan(NamedTuple):
    """The kernel operands of :func:`plan_groups` and their entry counts."""

    #: ``(array or deferred position, sample-last labels, read or None)``.
    kernels: tuple
    shapes: tuple
    labels: tuple
    #: Labels any kernel carries.
    carried: set
    #: Entries of every kernel, and of the views the range checks also read.
    elements: int
    read_elements: int
    #: Entries of the sample-last copies :func:`finish` makes.
    copies: int
    #: Entries of the largest array a range scan reads.
    scanned: int


def _pinned_labels(labels, classes):
    return tuple(label for label in labels if label and classes[label][0] == "pin")


def _read_classes(layout, dims, rank):
    """Label classes of the exclusions the ordered fold skips before it
    multiplies an operand of ``rank``: those of R batches up to ``rank``.

    A subset of the constraints of the full classes, so its pins never
    conflict where the full classes exist.
    """
    deltas = [(a, b) for a, b, batch in layout.skipped_deltas if batch <= rank]
    pins = [(label, value) for label, value, batch in layout.skipped_pins if batch <= rank]
    return label_classes(dims, deltas, pins)


def _plan_kernels(layout, dims, classes, samples):
    """The kernel operands in the sample-last layout :func:`finish` copies
    them into, with the views the ordered fold multiplies beyond them."""
    read_classes = {}
    kernels = []
    shapes = []
    read_sizes = []
    deferred_index = 0
    elements = copies = scanned = 0
    for (array, labels), rank in zip(layout.operands, layout.ranks):
        if array is None:
            kernels.append((deferred_index, (0,), None))
            shapes.append((samples,))
            deferred_index += 1
            elements += samples
            scanned = max(scanned, samples)
            continue
        read = None
        if rank not in read_classes:
            read_classes[rank] = _read_classes(layout, dims, rank)
        if _pinned_labels(labels, classes) != _pinned_labels(labels, read_classes[rank]):
            read = apply_classes(array, labels, read_classes[rank])[0]
            read_sizes.append(read.size)
            scanned = max(scanned, read.size)
        array, labels = apply_classes(array, labels, classes)
        shape, labels = sample_last(array.shape, labels)
        kernels.append((array, labels, read))
        shapes.append(shape)
        elements += array.size
        scanned = max(scanned, array.size)
        if array.ndim > 1 and labels[-1] == 0:
            copies += array.size
    kernel_labels = tuple(labels for _, labels, _ in kernels)
    return _KernelPlan(
        tuple(kernels),
        tuple(shapes),
        kernel_labels,
        set().union(*kernel_labels),
        elements,
        sum(read_sizes),
        copies,
        scanned,
    )


def _plan_group(operands, classes, summed, kernels):
    """``(resolved operands, plan, loop)`` of one coefficient group, or ``None``
    when a summed label class is carried by no operand or planning fails."""
    resolved = tuple(apply_classes(a, labels, classes) for a, labels in operands)
    if not summed <= kernels.carried.union(*(indices for _, indices in resolved)):
        return None
    plan, loop = planned_path(
        tuple(array.shape for array, _ in resolved) + kernels.shapes,
        tuple(indices for _, indices in resolved) + kernels.labels,
    )
    if plan is None:
        return None
    return resolved, plan, loop


def plan_groups(groups, layout, dims, samples):
    """Plan every coefficient group against the shared kernel operands.

    ``groups`` holds ``(phase, coefficient_operands)`` with operand labels
    over the summation labels of ``dims``.  Kernel operands are planned in
    the sample-last layout :func:`finish` copies them into.  A kernel that
    the full label classes slice more than the ordered fold does (a pin from
    a diagonal C table, or from an R support the ordered fold checks only
    after it has multiplied this kernel) also keeps ``read``, the view the
    ordered fold multiplies, for the range checks.  Returns ``(Contraction,
    classes)`` or ``None`` when the diagram vanishes structurally, a plan
    fails, or a summed label class is carried by no operand of a group: the
    ordered fold adds that group's product once per value of the label, and
    an index-free factor (the trace of an R without a partner label) or the
    scalar R product carries no label.  The cost covers the kernel passes
    and the planned steps; the factorized overhead adds the per-monomial
    work.  The workspace is
    :func:`~sft_wick._contraction_admission.finish_workspace`, with the
    results of every planned step of one group counted whether or not they
    are alive at once.
    """
    if layout.zero:
        return None
    classes = label_classes(dims, layout.deltas, layout.pins)
    if classes is None:
        return None
    kernels = _plan_kernels(layout, dims, classes, samples)
    summed = {value for kind, value in classes.values() if kind == "sum"}
    planned = []
    cost = kernel_cost(kernels.elements, kernels.read_elements, samples)
    # Bytes of one group's planned steps.
    intermediates = 0
    for phase, operands in groups:
        group = _plan_group(operands, classes, summed, kernels)
        if group is None:
            return None
        resolved, (steps, flops, step_elements), loop = group
        cost += step_cost(len(steps), flops)
        intermediates = max(intermediates, step_bytes(step_elements, loop))
        planned.append((phase, resolved, steps))
    if len(groups) > 1:
        # The accumulated result and the previous group's result.
        intermediates += 16 * samples
    deferred_peak = 0
    for kind, array in layout.deferred:
        cost += deferred_cost(kind, array.size)
        deferred_peak = max(deferred_peak, deferred_bytes(kind, array.size))
    workspace = finish_workspace(
        layout, samples, deferred_peak, kernels.copies, kernels.scanned, intermediates
    )
    return Contraction(tuple(planned), kernels.kernels, cost, workspace), classes


# ----------------------------------------------- numeric admission, run


def finish(contraction, layout, coefficient_ranges, assignments, *, phase_is_factor=True):
    """Numeric admission and execution of a planned contraction.

    ``coefficient_ranges`` holds, per group, the :func:`finite_range` of each
    coefficient operand (``None`` when one is not finite).  With
    ``phase_is_factor`` the group's phase counts as one more factor in the
    range rules.  ``"factorized"`` passes ``False`` and appends the ranges
    of the scalars its phase folds together (the prefactor and each
    rational), because the ordered preparation multiplies those one at a
    time and their product can be in range where a partial product is not.
    Returns the sample vector or ``None``.
    """
    if not supports_hold(layout.zero_checks):
        return None
    deferred = deferred_values(layout.deferred)
    if deferred is None:
        return None
    scanned = _scanned_kernels(contraction.kernels, deferred)
    if scanned is None:
        return None
    kernels, kernel_ranges = scanned
    if not products_admit(
        contraction.groups, coefficient_ranges, kernel_ranges, assignments, phase_is_factor
    ):
        return None
    return _run_groups(contraction.groups, kernels)


def _scanned_kernels(planned, deferred):
    """The kernel operands of a call and their :func:`finite_range`.

    Each kernel is copied into its sample-last layout and scanned; an
    index-free factor is taken from ``deferred``.  ``None`` when an entry is
    not finite.
    """
    kernels = []
    kernel_ranges = []
    for array, labels, read in planned:
        if type(array) is int:
            array = deferred[array]
        elif array.ndim > 1 and labels[-1] == 0:
            array = np.ascontiguousarray(np.moveaxis(array, 0, -1))
        scanned = finite_range(array)
        if scanned is None:
            return None
        if read is not None:
            # The ordered fold also multiplies these entries: by a diagonal C
            # table's zero, or before it reaches the R support that excludes
            # them.  inf or NaN there, or an overflowing product, gives NaN
            # or a floating-point event, so rules (1) and (2) cover them.
            # Rule (3) concerns the products the backend forms.
            covered = finite_range(read)
            if covered is None:
                return None
            scanned = (covered[0], scanned[1])
        kernel_ranges.append(scanned)
        kernels.append((array, labels))
    return kernels, kernel_ranges


def _run_groups(groups, kernels):
    """Contract every group with the kernels and add the group results."""
    result = None
    for phase, operands, steps in groups:
        value = execute((*operands, *kernels), steps)
        if phase != 1.0:
            np.multiply(value, phase, out=value)
        if result is None:
            result = value
        else:
            np.add(result, value, out=result)
    # Maps -0.0 to +0.0; the ordered fold starts from +0.0.  Only a negative
    # phase can produce -0.0 here: einsum accumulates into a zeroed output.
    np.add(result, 0.0, out=result)
    return result


# ------------------------------------------------------ shared call stage


class PlannedCall(NamedTuple):
    """A backend call that passed the planning stage of its admission."""

    layout: KernelLayout
    contraction: Contraction
    classes: dict
    #: :func:`restricted_supports` of the layout.
    restricted: tuple
    #: :func:`count_bytes` of the live-assignment count.
    counting: int
    #: Planned cost plus the backend's own work.
    cost: float


def plan_call(groups, batches, names, fixed, dims, extra, upper, mask_shape):
    """The planning stage both backends run after their lower-bound check.

    ``groups`` are the coefficient groups of :func:`plan_groups` and
    ``batches`` is ``(c_batches, r_batches, r_product, jacobians)``.
    Resolves the kernel layout, plans the groups, applies the cost rule to
    the planned cost plus the backend's ``extra`` work against ``upper``
    (the ordered estimate with every assignment live), and sizes the
    live-assignment count over a coefficient mask of ``mask_shape``
    (``None`` for no mask).  Returns a :class:`PlannedCall` or ``None``; the
    caller then applies its own workspace rule.
    """
    c_batches, r_batches, r_product, jacobians = batches
    layout = kernel_layout(c_batches, r_batches, r_product, jacobians, names, fixed)
    if layout is None:
        return None
    planned = plan_groups(groups, layout, dims, r_product.size)
    if planned is None:
        return None
    contraction, classes = planned
    cost = contraction.cost + extra
    if not backend_is_cheaper(cost, upper):
        return None
    restricted = restricted_supports(layout.supports)
    counting = count_bytes(dims, mask_shape, restricted)
    if counting is None:
        return None
    return PlannedCall(layout, contraction, classes, restricted, counting, cost)


def live_cost_admits(planned, assignments, nonzero, live, counted, samples) -> bool:
    """The cost rule once the live assignments are counted: the planned cost
    plus the count's own cost (``counted``) against the ordered estimate."""
    layout = planned.layout
    estimate = ordered_cost(assignments, nonzero, live, layout.n_c, layout.n_r, samples)
    return backend_is_cheaper(planned.cost + counted, estimate)


# --------------------------------------------------------- the backend


def _coefficient_labels(coeff, prop_idx):
    """The axes of ``prop_idx`` when ``coeff`` spans them, or ``None``.

    ``coeff`` must be a float64 or complex128 array of the propagator
    indices' shape with more than one entry.
    """
    if not coupling_array_ok(coeff):
        return None
    names = label_axes(prop_idx)
    if names is None:
        return None
    if coeff.shape != tuple(dim for _, dim in names.values()) or coeff.size <= 1:
        return None
    return names


def try_static_contraction(
    coeff,
    prop_idx,
    fixed_indices,
    c_batches,
    r_product,
    jacobians,
    r_batches,
    e_psi,
    max_bytes,
):
    """The ``"einsum"`` backend: contract the prepared coefficient array.

    Returns a float64 sample vector under the accuracy and behaviour
    contract of :mod:`sft_wick._contraction_admission`, or ``None`` to keep
    the ordered fold.  No input is modified or retained.
    """
    if not call_inputs_ok(max_bytes, fixed_indices, c_batches, r_batches, r_product):
        return None
    names = _coefficient_labels(coeff, prop_idx)
    if names is None:
        return None
    dims = {axis: dim for axis, dim in names.values()}
    samples = r_product.size
    n_c, n_r = len(c_batches), len(r_batches)
    extra = coefficient_cost(coeff.size)
    # The ordered estimate with every assignment live bounds it from above;
    # the step count bounds the backend from below.  Small diagrams decline
    # here, before any layout work.
    upper = ordered_cost(coeff.size, coeff.size, coeff.size, n_c, n_r, samples)
    if not backend_is_cheaper(minimum_backend_cost(2 + n_c + n_r) + extra, upper):
        return None
    planned = plan_call(
        ((1.0, ((coeff, tuple(dims)),)),),
        (c_batches, r_batches, r_product, jacobians),
        names,
        fixed_indices,
        dims,
        extra,
        upper,
        coeff.shape,
    )
    if planned is None:
        return None
    workspace = einsum_workspace(
        coeff.size, planned.contraction, planned.layout, planned.counting
    )
    if workspace > max_bytes:
        return None
    return _contract_rotated(planned, coeff, e_psi, dims, samples)


def _contract_rotated(planned, coeff, e_psi, dims, samples):
    """The numeric stage of an ``"einsum"`` call: the phase rule, the
    live-assignment count and its cost check, then :func:`finish`."""
    rotated = rotated_real_coefficient(coeff, e_psi)
    if rotated is None:
        return None
    mask = (rotated != 0.0).astype(np.float64)
    live, counted = live_assignments(dims, mask, (), restricted=planned.restricted)
    nonzero = int(np.count_nonzero(mask))
    del mask
    if live is None:
        return None
    if not live_cost_admits(planned, coeff.size, nonzero, live, counted, samples):
        return None
    coefficient_range = finite_range(rotated)
    ((_, _, steps),) = planned.contraction.groups
    operand = apply_classes(rotated, tuple(dims), planned.classes)
    contraction = replace(planned.contraction, groups=((1.0, (operand,), steps),))
    return finish(contraction, planned.layout, ((coefficient_range,),), coeff.size)
