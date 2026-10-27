"""Shared admission, planning and accuracy contract of the opt-in backends.

The static integrand of a diagram is, per sample ``s``,

    values[s] = sum_a c(a) * prod_k K_k(s, a),

a sum over the component assignments ``a`` of the coefficient ``c`` (the
rotated real part of the coupling array, or for ``"factorized"`` a sum of
monomials of the original coupling tensors) times the C and R propagator
batches, the scalar R product and the Jacobian.  The default ``"ordered"``
backend folds the assignments one at a time.  ``"einsum"`` and
``"factorized"`` evaluate the same sum as a sequence of pairwise
``numpy.einsum`` contractions.  An opt-in backend either returns a result or
declines with ``None``; the caller then runs the ordered fold.

Accuracy contract.  Let ``S[s] = sum_a sum_t |c_t(a) prod_k K_k(s, a)|``
(``t`` runs over the ``T`` monomials, ``T = 1`` for ``"einsum"``), ``A`` the
number of component assignments, ``m`` the number of factors in one product
and ``P = prod_k max(1, max|K_k|)`` with the coefficient factors included.
With ``u = 2**-53``, a returned result satisfies, at every sample,

    |result - exact| <= 1.01 * (m + A*T) * u * S + m*A*T * 2**-1074 * P,

where ``exact`` is the sum computed in exact arithmetic from the same
float64 inputs.  The ordered fold satisfies the same bound; the two results
generally differ in their last bits.  The second term bounds gradual
underflow and is negligible unless the inputs approach 1e-300.

Behaviour contract.  A backend only returns a result when (1) every array it
reads is finite, (2) no product or partial sum can overflow in any
evaluation order, (3) every coefficient passes the phase rule of
:func:`sft_wick.evaluate._real_or_raise`, (4) ``numpy.geterr()["under"]`` is
``"ignore"`` (``numpy.einsum`` reports no floating-point errors, so a
stricter underflow policy would see different events), (5) the planned
array workspace fits ``contraction_max_bytes`` and (6) the cost model
predicts it to be cheaper than the ordered fold.  Under (1) and (2) neither
evaluation order produces inf, NaN or a floating-point error, so an
exception from the ordered fold (phase error, non-isotropic index-free C)
is reproduced by declining.  The output never holds ``-0.0``.

The backends accept the package's own classes and exact float64 or
complex128 arrays only.  Replacing package methods or helpers at run time
(monkeypatching) is unsupported and is not detected.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import math

import numpy as np

#: Default workspace budget, in bytes, for one coupling contraction (64 MiB).
DEFAULT_CONTRACTION_MAX_BYTES = 64 * 1024 * 1024

#: Largest binary exponent a product or partial sum may reach.  Four bits
#: below the float64 limit leave room for rounding growth.
_MAX_EXPONENT = 1020

#: Coefficients whose rotated imaginary part is nonzero are checked one by
#: one with the ordered projector; above this count the backend declines.
_MAX_PHASE_CHECKS = 4096

_QUARTER_TURNS = (1 + 0j, 1j, -1 + 0j, -1j)

#: Operands of one planned product; the pairwise search costs O(k**3).
_MAX_OPERANDS = 64

# Cost model, in units of about one nanosecond.  The constants were fitted to
# timings of 275 captured diagram calls (N = 2, 3, 5; 256 to 65536 samples;
# Python 3.10 with NumPy 2.2 and Python 3.14 with NumPy 2.4).  Ordered
# constants are about 0.65 of the fitted values and backend constants about
# 1.6 times, so an admitted contraction is faster than the ordered fold with
# a margin; no admitted call in the calibration set was slower.
#: Ordered fold: Python work for an assignment whose coefficient is zero.
_ORDERED_ZERO = 200.0
#: Ordered fold: Python work for an assignment with a nonzero coefficient
#: (phase projection, index map, R support lookup).
_ORDERED_ASSIGNMENT = 1500.0
#: Ordered fold: fixed cost of one NumPy call on a sample vector.
_ORDERED_CALL = 200.0
#: Ordered fold: cost per sample of one NumPy call.
_ORDERED_ELEMENT = 0.3
#: Backend: fixed cost of one call (layout, label classes, plan lookups).
_BACKEND_CALL = 20000.0
#: Backend: cost per kernel element of one pass (layout copy, two scans).
_BACKEND_ELEMENT = 0.7
#: Backend (einsum): cost per coefficient element (rotation, masks, scan).
_BACKEND_COEFFICIENT = 60.0
#: Backend: fixed cost of one pairwise contraction step.
_BACKEND_STEP = 20000.0
#: Backend: cost of one floating-point operation in a contraction step.
_BACKEND_FLOP = 0.25
#: Backend (factorized): extra fixed cost of one call.
_FACTORIZED_CALL = 40000.0
#: Backend (factorized): fixed cost per monomial beyond its steps.
_BACKEND_MONOMIAL = 5000.0
#: Backend (factorized): cost per monomial and sample (phase, accumulation).
_BACKEND_MONOMIAL_SAMPLE = 5.0


def validate_contraction_options(backend, max_bytes) -> None:
    """Validate the backend selector and the positive workspace budget."""
    if type(backend) is not str or backend not in ("ordered", "einsum", "factorized"):
        raise ValueError(
            "contraction_backend must be 'ordered', 'einsum' or 'factorized'"
        )
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("contraction_max_bytes must be a positive integer")


def _native_dtype(array, scalar_type) -> bool:
    # dtype equality/str aliases longdouble and float64 on macOS ARM.
    return (
        array.dtype.type is scalar_type
        and array.dtype.isnative
        and array.dtype.metadata is None
    )


def _float64(array) -> bool:
    return type(array) is np.ndarray and _native_dtype(array, np.float64)


def _ordinary_label(value) -> bool:
    # Numeric literals ("1") resolve differently in the ordered selector.
    return type(value) is str and value.isidentifier()


def errors_allow_backend() -> bool:
    """Whether the floating-point error policy lets a backend run."""
    return np.geterr()["under"] == "ignore"


def finite_maximum(array) -> float | None:
    """``max|array|`` when every entry is finite, otherwise ``None``.

    ``max`` and ``min`` propagate NaN and infinities, so two reductions decide
    finiteness without a temporary array.
    """
    if array.size == 0:
        return 0.0
    with np.errstate(all="ignore"):
        high = float(np.max(array))
        low = float(np.min(array))
    if not (math.isfinite(high) and math.isfinite(low)):
        return None
    return max(abs(high), abs(low))


def exponent_bound(maximum: float) -> int:
    """Smallest ``e >= 0`` with ``maximum < 2**e`` (0 for ``maximum <= 1/2``)."""
    if maximum == 0.0:
        return 0
    return max(0, math.frexp(maximum)[1])


def rotated_real_coefficient(coeff, e_psi):
    """The real coefficient the ordered fold uses at every assignment.

    The ordered fold multiplies by ``_real_or_raise(c, e_psi)``: the real part
    of ``c * 1j**e_psi``, accepted when the imaginary part is zero, below the
    floor, or within tolerance.  That real part is ``(a, -b, -a, b)[e_psi % 4]``
    for ``c = a + 1j*b``.  Entries with a nonzero rotated imaginary part are
    tested with the ordered projector itself.  Returns a new contiguous
    float64 array, or ``None`` when an entry would make the ordered fold raise
    or the phase is not an exact quarter turn.
    """
    if type(e_psi) is not int or not 0 <= e_psi <= 1024:
        return None
    if (1j) ** e_psi not in _QUARTER_TURNS:
        return None
    quarter = e_psi % 4
    sign = 1.0 if quarter in (0, 3) else -1.0
    if _native_dtype(coeff, np.float64):
        if quarter % 2:
            return None
        return np.multiply(coeff, sign)
    if not _native_dtype(coeff, np.complex128):
        return None
    kept, other = (coeff.real, coeff.imag) if quarter % 2 == 0 else (
        coeff.imag,
        coeff.real,
    )
    if np.any(other):
        positions = np.flatnonzero(other)
        if positions.size > _MAX_PHASE_CHECKS:
            return None
        from .evaluate import _real_or_raise

        for position in positions:
            try:
                _real_or_raise(coeff.flat[position], e_psi)
            except ValueError:
                return None
    return np.multiply(kept, sign)


@dataclass(frozen=True)
class KernelLayout:
    """The propagator side of a contraction, as einsum operands.

    Labels are integers: ``0`` is the sample axis and ``1..k`` the summation
    indices.  ``operands`` holds ``(array, labels)`` pairs; an entry whose
    array is ``None`` is an index-free factor computed later from ``deferred``
    (``("isotropic", C)`` or ``("trace", R)``), always with labels ``(0,)``.
    ``deltas`` lists label pairs a structurally diagonal factor sets equal and
    ``pins`` labels it fixes to a component; ``zero`` marks a diagonal factor
    between two different fixed components.  ``supports`` keeps each resolved
    R support with its labels for the ordered cost estimate, and
    ``zero_checks`` the R entries outside their support, which must vanish.
    """

    operands: tuple
    deferred: tuple
    deltas: tuple
    pins: tuple
    zero: bool
    supports: tuple
    zero_checks: tuple
    n_c: int
    n_r: int


def _resolve(label, dim, names, fixed):
    """("sum", axis) for a summation label, ("pin", value) for a fixed one."""
    if not _ordinary_label(label):
        return None
    if label in names:
        axis, size = names[label]
        return ("sum", axis) if size == dim else None
    value = fixed.get(label)
    if type(value) is int and 0 <= value < dim:
        return ("pin", value)
    return None


def _select(array, left, right):
    """A view of an ``(n, N, N)`` batch with fixed components sliced away."""
    index = [slice(None)]
    labels = [0]
    for kind, value in (left, right):
        if kind == "sum":
            index.append(slice(None))
            labels.append(value)
        else:
            index.append(value)
    return array[tuple(index)], tuple(labels)


def _diagonal(vectors, left, right, operands, deltas, pins) -> bool:
    """Add a diagonal factor ``delta_ab v[:, a]``; return True if it is zero."""
    if left[0] == "sum" and right[0] == "sum":
        operands.append((vectors, (0, left[1])))
        if left[1] != right[1]:
            deltas.append((left[1], right[1]))
        return False
    if left[0] == "pin" and right[0] == "pin":
        if left[1] != right[1]:
            return True
        operands.append((vectors[:, left[1]], (0,)))
        return False
    free, pinned = (left, right) if left[0] == "sum" else (right, left)
    operands.append((vectors[:, pinned[1]], (0,)))
    pins.append((free[1], pinned[1]))
    return False


def kernel_layout(c_batches, r_batches, r_product, jacobians, names, fixed):
    """Resolve the C and R batches of one integrand call, or return ``None``.

    Reproduces the ordered selectors: summation names take precedence over
    fixed indices; a ``(n, N)`` C batch is ``delta_ab C[:, a]``; an
    index-free C is :func:`~sft_wick.evaluate._isotropic_C`; an R with a
    missing label is its trace; an R entry outside its support contributes
    nothing.  An R whose support lies on the diagonal is read through its
    diagonal only.  No float array is read here except views and the boolean
    supports.
    """
    if (
        type(c_batches) not in (list, tuple)
        or type(r_batches) not in (list, tuple)
        or not _float64(r_product)
        or r_product.ndim != 1
        or not r_product.size
        or not _float64(jacobians)
        or jacobians.shape != r_product.shape
    ):
        return None
    samples = r_product.size
    operands = [(r_product, (0,)), (jacobians, (0,))]
    deferred = []
    deltas = []
    pins = []
    supports = []
    zero_checks = []
    zero = False
    for batch in c_batches:
        if type(batch) is not tuple or len(batch) != 3:
            return None
        array, left, right = batch
        if (
            not _float64(array)
            or array.ndim not in (2, 3)
            or array.shape[0] != samples
            or not array.shape[1]
            or (array.ndim == 3 and array.shape[2] != array.shape[1])
        ):
            return None
        if left is None and right is None:
            operands.append((None, (0,)))
            deferred.append(("isotropic", array))
            continue
        dim = array.shape[1]
        a = _resolve(left, dim, names, fixed)
        b = _resolve(right, dim, names, fixed)
        if a is None or b is None:
            return None
        if array.ndim == 3:
            operands.append(_select(array, a, b))
        else:
            zero |= _diagonal(array, a, b, operands, deltas, pins)
    for batch in r_batches:
        if type(batch) is not tuple or len(batch) != 4:
            return None
        array, left, right, support = batch
        if (
            not _float64(array)
            or array.ndim != 3
            or array.shape[0] != samples
            or not array.shape[1]
            or array.shape[2] != array.shape[1]
        ):
            return None
        dim = array.shape[1]
        if (
            type(support) is not np.ndarray
            or not _native_dtype(support, np.bool_)
            or support.shape != (dim, dim)
        ):
            return None
        if not (left and right):
            operands.append((None, (0,)))
            deferred.append(("trace", array))
            continue
        a = _resolve(left, dim, names, fixed)
        b = _resolve(right, dim, names, fixed)
        if a is None or b is None:
            return None
        on_diagonal = np.diagonal(support)
        supports.append((support, a, b))
        if np.count_nonzero(support) == np.count_nonzero(on_diagonal):
            diagonal = np.diagonal(array, axis1=1, axis2=2)
            zero |= _diagonal(diagonal, a, b, operands, deltas, pins)
            for k in np.flatnonzero(~on_diagonal):
                zero_checks.append(diagonal[:, k])
        else:
            operands.append(_select(array, a, b))
            if not support.all():
                zero_checks.append((array, ~support))
    return KernelLayout(
        tuple(operands),
        tuple(deferred),
        tuple(deltas),
        tuple(pins),
        zero,
        tuple(supports),
        tuple(zero_checks),
        len(c_batches),
        len(r_batches),
    )


def label_classes(dims, deltas, pins):
    """Map each summation label to ``("sum", representative)`` or ``("pin", value)``.

    ``dims`` maps labels to sizes.  Returns ``None`` when two pins of one
    class disagree (the diagram vanishes structurally).
    """
    parent = {label: label for label in dims}

    def find(label):
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    for left, right in deltas:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)
    pinned = {}
    for label, value in pins:
        root = find(label)
        if pinned.setdefault(root, value) != value:
            return None
    return {
        label: ("pin", pinned[find(label)])
        if find(label) in pinned
        else ("sum", find(label))
        for label in dims
    }


def apply_classes(array, labels, classes):
    """Slice pinned axes and rename merged labels of one operand."""
    index = []
    renamed = []
    sliced = False
    for label in labels:
        if label == 0:
            index.append(slice(None))
            renamed.append(0)
            continue
        kind, value = classes[label]
        if kind == "pin":
            index.append(value)
            sliced = True
        else:
            index.append(slice(None))
            renamed.append(value)
    if sliced:
        array = array[tuple(index)]
    return array, tuple(renamed)


def sample_last(shape, labels):
    """Shape and labels of an operand with its sample axis moved last."""
    if 0 not in labels or labels[-1] == 0:
        return shape, labels
    axis = labels.index(0)
    return (
        shape[:axis] + shape[axis + 1 :] + (shape[axis],),
        labels[:axis] + labels[axis + 1 :] + (0,),
    )


@lru_cache(maxsize=4096)
def plan_contraction(shapes, labels):
    """Pairwise contraction path of one product sum, from shapes only.

    At each step the pair with the smallest loop space (then the smallest
    result) is contracted; labels no other operand carries are summed.
    Returns ``(steps, flops, workspace)`` or ``None``: ``steps`` holds
    ``(i, j, kept)`` with the positions of the two operands in the current
    list (the result is appended) and the result labels, ordered with the
    sample label ``0`` last; ``flops`` counts two operations per point of
    every step's loop space; ``workspace`` counts every step's result
    elements.  The final operand must carry only the sample label.
    """
    if len(shapes) > _MAX_OPERANDS:
        return None
    dims = {}
    for shape, indices in zip(shapes, labels):
        if len(shape) != len(indices):
            return None
        for label, size in zip(indices, shape):
            if dims.setdefault(label, size) != size:
                return None
    sets = [frozenset(indices) for indices in labels]
    counts = {}
    for indices in sets:
        for label in indices:
            counts[label] = counts.get(label, 0) + 1
    steps = []
    flops = workspace = 0
    while len(sets) > 1:
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
                    math.prod(dims[label] for label in involved),
                    math.prod(dims[label] for label in kept),
                    i,
                    j,
                )
                if best is None or key < best[0]:
                    best = (key, kept)
        (loop, out, i, j), kept = best
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


def ordered_cost(assignments, nonzero, live, n_c, n_r, samples) -> float:
    """Estimated cost of the ordered fold for one call.

    Every assignment costs a Python iteration; a nonzero coefficient adds the
    phase projection and R lookups; a live assignment (nonzero coefficient,
    every R entry supported) runs ``n_r + n_c + 5`` vector operations.
    """
    calls = live * (n_c + n_r + 5)
    return (
        assignments * _ORDERED_ZERO
        + nonzero * _ORDERED_ASSIGNMENT
        + calls * (_ORDERED_CALL + samples * _ORDERED_ELEMENT)
    )


def backend_is_cheaper(backend: float, ordered: float) -> bool:
    """The cost rule: run a backend only when it is estimated to be cheaper."""
    return backend < ordered


def live_assignments(dims, coefficient_mask, supports):
    """Number of assignments the ordered fold multiplies out.

    ``dims`` maps the summation labels to sizes, ``coefficient_mask`` is a
    0/1 float array over all of them in order (or ``None`` for a dense
    coefficient) and ``supports`` the resolved R supports.  The count is
    exact (float64 holds integers below 2**53).
    """
    arguments = []
    used = set()
    if coefficient_mask is not None:
        arguments.extend((coefficient_mask, list(dims)))
        used.update(dims)
    for support, left, right in supports:
        index = []
        indices = []
        for kind, value in (left, right):
            if kind == "sum":
                index.append(slice(None))
                indices.append(value)
            else:
                index.append(value)
        arguments.extend((support[tuple(index)].astype(np.float64), indices))
        used.update(indices)
    free = math.prod(size for label, size in dims.items() if label not in used)
    if not arguments:
        return free
    arguments.append([])
    # A direct loop needs no path search; many operands need a pairwise path.
    optimize = "greedy" if len(arguments) > 60 else False
    return free * int(np.einsum(*arguments, optimize=optimize))


def _deferred_values(deferred):
    """Compute index-free factors with the ordered fold's own calls.

    The ordered fold may never reach these factors, so they are computed
    without floating-point events; a non-finite result declines later.
    """
    from .evaluate import _isotropic_C

    values = []
    with np.errstate(all="ignore"):
        for kind, array in deferred:
            if kind == "isotropic":
                try:
                    values.append(_isotropic_C(array, batched=True))
                except ValueError:
                    return None
            else:
                values.append(np.einsum("iaa->i", array))
    return values


def _supports_hold(zero_checks) -> bool:
    for check in zero_checks:
        if type(check) is tuple:
            array, outside = check
            if np.any(array[:, outside]):
                return False
        elif np.any(check):
            return False
    return True


@dataclass(frozen=True)
class Contraction:
    """A planned contraction of one or more coefficient groups."""

    groups: tuple  # ((phase, ((array, labels), ...), steps), ...)
    kernels: tuple  # ((array or deferred position, sample-last labels), ...)
    cost: float
    workspace: int


def plan_groups(groups, layout, dims, samples):
    """Plan every coefficient group against the shared kernel operands.

    ``groups`` holds ``(phase, coefficient_operands)`` with operand labels
    over the summation labels of ``dims``.  Kernel operands are planned in
    the sample-last layout :func:`finish` copies them into.  Returns
    ``(Contraction, classes)`` or ``None`` when the diagram vanishes
    structurally or a plan fails.
    """
    if layout.zero:
        return None
    classes = label_classes(dims, layout.deltas, layout.pins)
    if classes is None:
        return None
    kernels = []
    shapes = []
    deferred_index = 0
    for array, labels in layout.operands:
        if array is None:
            kernels.append((deferred_index, (0,)))
            shapes.append((samples,))
            deferred_index += 1
            continue
        array, labels = apply_classes(array, labels, classes)
        shape, labels = sample_last(array.shape, labels)
        kernels.append((array, labels))
        shapes.append(shape)
    kernel_shapes = tuple(shapes)
    kernel_labels = tuple(labels for _, labels in kernels)
    planned = []
    elements = sum(math.prod(shape) for shape in kernel_shapes)
    # Copies into the sample-last layout, two range scans, output fix-up.
    cost = _BACKEND_CALL + 3 * elements * _BACKEND_ELEMENT + samples
    workspace = 0
    for phase, operands in groups:
        resolved = tuple(apply_classes(a, labels, classes) for a, labels in operands)
        plan = plan_contraction(
            tuple(array.shape for array, _ in resolved) + kernel_shapes,
            tuple(indices for _, indices in resolved) + kernel_labels,
        )
        if plan is None:
            return None
        steps, flops, step_elements = plan
        cost += len(steps) * _BACKEND_STEP + flops * _BACKEND_FLOP
        if len(groups) > 1:
            cost += _BACKEND_MONOMIAL + _BACKEND_MONOMIAL_SAMPLE * samples
        workspace = max(workspace, step_elements)
        planned.append((phase, resolved, steps))
    for kind, array in layout.deferred:
        cost += array.size * (5 if kind == "isotropic" else 1) * _BACKEND_ELEMENT
        workspace += 3 * array.size if kind == "isotropic" else samples
    for check in layout.zero_checks:
        if type(check) is tuple:
            workspace += samples * int(np.count_nonzero(check[1]))
    # Sample-last kernel copies, the accumulator and one group result.
    workspace += elements + samples * (2 if len(groups) > 1 else 1)
    return Contraction(tuple(planned), tuple(kernels), cost, 8 * workspace), classes


def finish(contraction, layout, coefficient_maxima, assignments):
    """Numeric admission and execution of a planned contraction.

    ``coefficient_maxima`` holds, per group, the maxima of its coefficient
    operands (``None`` when one is not finite).  Returns the sample vector or
    ``None``.
    """
    if not _supports_hold(layout.zero_checks):
        return None
    deferred = _deferred_values(layout.deferred)
    if deferred is None:
        return None
    kernels = []
    kernel_exponent = 0
    for array, labels in contraction.kernels:
        if type(array) is int:
            array = deferred[array]
        elif array.ndim > 1 and labels[-1] == 0:
            array = np.ascontiguousarray(np.moveaxis(array, 0, -1))
        maximum = finite_maximum(array)
        if maximum is None:
            return None
        kernel_exponent += exponent_bound(maximum)
        kernels.append((array, labels))
    headroom = (assignments - 1).bit_length() + (len(contraction.groups) - 1).bit_length()
    for (phase, _, _), maxima in zip(contraction.groups, coefficient_maxima):
        if any(maximum is None for maximum in maxima):
            return None
        total = headroom + kernel_exponent + exponent_bound(abs(phase))
        total += sum(exponent_bound(maximum) for maximum in maxima)
        if total > _MAX_EXPONENT:
            return None
    result = None
    for phase, operands, steps in contraction.groups:
        value = execute((*operands, *kernels), steps)
        if phase != 1.0:
            np.multiply(value, phase, out=value)
        if result is None:
            result = value
        else:
            np.add(result, value, out=result)
    # Maps -0.0 to +0.0; the ordered fold starts from +0.0.
    np.add(result, 0.0, out=result)
    return result


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

    Returns a float64 sample vector under the module's accuracy and
    behaviour contract, or ``None`` to keep the ordered fold.  No input is
    modified or retained.
    """
    if type(max_bytes) is not int or max_bytes <= 0 or not errors_allow_backend():
        return None
    if (
        type(coeff) is not np.ndarray
        or not (_native_dtype(coeff, np.float64) or _native_dtype(coeff, np.complex128))
        or type(prop_idx) is not tuple
        or not 1 <= len(prop_idx) <= 51
        or type(fixed_indices) is not dict
        or any(
            not _ordinary_label(k) or type(v) is not int
            for k, v in fixed_indices.items()
        )
    ):
        return None
    names = {}
    for axis, item in enumerate(prop_idx, 1):
        if (
            type(item) is not tuple
            or len(item) != 2
            or not _ordinary_label(item[0])
            or type(item[1]) is not int
            or item[1] <= 0
            or item[0] in names
        ):
            return None
        names[item[0]] = axis, item[1]
    dims = {axis: dim for axis, dim in names.values()}
    if coeff.shape != tuple(dims.values()) or coeff.size <= 1:
        return None
    layout = kernel_layout(c_batches, r_batches, r_product, jacobians, names, fixed_indices)
    if layout is None:
        return None
    samples = r_product.size
    planned = plan_groups(((1.0, ((coeff, tuple(dims)),)),), layout, dims, samples)
    if planned is None:
        return None
    contraction, classes = planned
    # The rotated coefficient and its zero mask.
    if contraction.workspace + 16 * coeff.size > max_bytes:
        return None
    rotated = rotated_real_coefficient(coeff, e_psi)
    if rotated is None:
        return None
    mask = (rotated != 0.0).astype(np.float64)
    estimate = ordered_cost(
        coeff.size,
        int(np.count_nonzero(mask)),
        live_assignments(dims, mask, layout.supports),
        layout.n_c,
        layout.n_r,
        samples,
    )
    backend = contraction.cost + coeff.size * _BACKEND_COEFFICIENT
    if not backend_is_cheaper(backend, estimate):
        return None
    ((_, _, steps),) = contraction.groups
    operand = apply_classes(rotated, tuple(dims), classes)
    contraction = replace(contraction, groups=((1.0, (operand,), steps),))
    return finish(contraction, layout, ((finite_maximum(rotated),),), coeff.size)
