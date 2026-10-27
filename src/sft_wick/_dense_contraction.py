"""Opt-in, conservatively admitted static component contractions.

Reassociation changes floating-point rounding.  Admission does not certify a
caller-chosen accuracy tolerance: complete scientific results must be compared
under an explicitly declared tolerance.  ``None`` asks the caller to retain its
ordered evaluator and its existing diagnostics, callback and selector rules.

The byte limit bounds conservative *planned ndarray payload*, including borrowed
inputs and possible copies; it is not a bound on process RSS, native allocator
overhead, or private BLAS workspace.  Plans retain shape/index metadata only.
"""

from __future__ import annotations

from functools import lru_cache
import math

import numpy as np

#: Default workspace budget, in bytes, for one coupling contraction (64 MiB).
DEFAULT_CONTRACTION_MAX_BYTES = 64 * 1024 * 1024

_DEFAULT_ERRORS = {
    "divide": "warn",
    "over": "warn",
    "under": "ignore",
    "invalid": "warn",
}
_SCAN_ELEMENTS = 4096
_SCAN_RESERVE = 64 * _SCAN_ELEMENTS


def validate_contraction_options(backend, max_bytes) -> None:
    """Validate the opt-in selector and conservative ndarray-payload limit."""
    if type(backend) is not str or backend not in ("ordered", "einsum", "factorized"):
        raise ValueError(
            "contraction_backend must be 'ordered', 'einsum' or 'factorized'"
        )
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("contraction_max_bytes must be a positive integer")


def _native_dtype(array, scalar_type):
    # dtype equality/str aliases longdouble and float64 on macOS ARM.
    return (
        array.dtype.type is scalar_type
        and array.dtype.isnative
        and array.dtype.metadata is None
    )


def _ordinary_label(value) -> bool:
    # In particular reject numeric literals even when explicitly pinned.  The
    # existing expression and propagator selectors have different literal rules.
    return type(value) is str and value.isidentifier()


def _scan(array):
    """Bounded-buffer inspection of a native real array, including strided views."""
    maximum = 0.0
    minimum = math.inf
    count = 0
    iterator = np.nditer(
        array,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        order="K",
        buffersize=_SCAN_ELEMENTS,
    )
    for chunk in iterator:
        if not bool(np.isfinite(chunk).all()):
            return None
        absolute = np.abs(chunk)
        maximum = max(maximum, float(np.max(absolute, initial=0.0)))
        nonzero = absolute[absolute != 0.0]
        count += int(nonzero.size)
        if nonzero.size:
            minimum = min(minimum, float(np.min(nonzero)))
    return maximum, minimum, count


def _phase_component(coeff, e_psi):
    """Choose the real rotated component without allocating a complex rotation."""
    if type(e_psi) is not int or not 0 <= e_psi <= 1024:
        return None
    # This is the exact scalar-projector phase, not a replacement modular rule.
    phase = 1j**e_psi
    if phase not in (1 + 0j, -1 + 0j, 1j, -1j):
        return None
    if _native_dtype(coeff, np.float64):
        if phase.imag:
            return None
        return coeff, phase.real
    real_scan = _scan(coeff.real)
    imag_scan = _scan(coeff.imag)
    if real_scan is None or imag_scan is None:
        return None
    if phase.imag:
        if real_scan[2]:
            return None
        return coeff.imag, -phase.imag
    if imag_scan[2]:
        return None
    return coeff.real, phase.real


@lru_cache(maxsize=32)
def _plan(shapes, labels):
    """Return a shape-only path and exact integer intermediate payload counts."""
    scalar = np.ones((), dtype=np.float64)
    arguments = []
    dimensions = {}
    for shape, indices in zip(shapes, labels):
        arguments.extend((np.broadcast_to(scalar, shape), list(indices)))
        for index, dimension in zip(indices, shape):
            old = dimensions.setdefault(index, dimension)
            if old != dimension:
                return None
    arguments.append([0])
    try:
        path = np.einsum_path(*arguments, optimize="greedy")[0]
    except (ValueError, TypeError):
        return None
    remaining = [set(indices) for indices in labels]
    total = maximum = 0
    for step in path[1:]:
        if (
            type(step) is not tuple
            or not step
            or len(set(step)) != len(step)
            or any(type(i) is not int or not 0 <= i < len(remaining) for i in step)
        ):
            return None
        selected = set().union(*(remaining[i] for i in step))
        others = {0}
        others.update(*(remaining[i] for i in range(len(remaining)) if i not in step))
        result = selected & others
        elements = math.prod(dimensions[i] for i in result)
        total += elements
        maximum = max(maximum, elements)
        remaining = [indices for i, indices in enumerate(remaining) if i not in step]
        remaining.append(result)
    if remaining != [{0}]:
        return None
    return tuple(path), total, maximum


def _factor_metadata(batch, names, fixed, samples, *, response):
    if type(batch) is not tuple or len(batch) != (4 if response else 3):
        return None
    array, left, right = batch[:3]
    if (
        type(array) is not np.ndarray
        or not _native_dtype(array, np.float64)
        or array.ndim != 3
        or array.shape[0] != samples
        or array.shape[1] != array.shape[2]
        or not array.shape[1]
        or not _ordinary_label(left)
        or not _ordinary_label(right)
    ):
        return None
    n = array.shape[1]
    slices = [slice(None)]
    indices = [0]
    shape = [samples]
    for label in (left, right):
        # The ordered merge is {**fixed, **assignment}; summation names win.
        if label in names:
            axis, dimension = names[label]
            if dimension != n:
                return None
            slices.append(slice(None))
            indices.append(axis)
            shape.append(n)
        elif label in fixed and type(fixed[label]) is int and 0 <= fixed[label] < n:
            slices.append(fixed[label])
        else:
            return None
    support = None
    if response:
        support = batch[3]
        if (
            type(support) is not np.ndarray
            or not _native_dtype(support, np.bool_)
            or support.shape != (n, n)
        ):
            return None
    return array, tuple(slices), tuple(indices), tuple(shape), support


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
    """Return an admitted float64 sample vector, otherwise ``None``.

    Full matrix C/R factors, ordinary resolved labels, pins and repeated labels
    are supported.  Scalar, sparse, trace, diagonal-batch, invalid-phase and
    numerical-range cases retain the ordered evaluator.  The caller must guard
    its concrete integrand/term/helper identities before invoking this helper.
    No input is modified or retained in the bounded structural plan cache.
    """
    if type(max_bytes) is not int or max_bytes <= 0 or np.geterr() != _DEFAULT_ERRORS:
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
        or type(c_batches) not in (tuple, list)
        or type(r_batches) not in (tuple, list)
        or not c_batches
        or type(r_product) is not np.ndarray
        or not _native_dtype(r_product, np.float64)
        or r_product.ndim != 1
        or not r_product.size
        or type(jacobians) is not np.ndarray
        or not _native_dtype(jacobians, np.float64)
        or jacobians.shape != r_product.shape
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
    shape = tuple(dimension for _, dimension in prop_idx)
    if coeff.shape != shape or coeff.size <= 1:
        return None
    samples = r_product.size
    factors = []
    for batches, response in ((c_batches, False), (r_batches, True)):
        for batch in batches:
            item = _factor_metadata(
                batch, names, fixed_indices, samples, response=response
            )
            if item is None:
                return None
            factors.append(item)
    # Avoid backend argument-limit failures before inspecting numeric inputs.
    if len(factors) + 3 > 31:
        return None
    shapes = (shape, *(item[3] for item in factors), (samples,), (samples,))
    labels = (
        tuple(range(1, len(prop_idx) + 1)),
        *(item[2] for item in factors),
        (0,),
        (0,),
    )
    planned = _plan(shapes, labels)
    if planned is None:
        return None
    path, intermediate_elements, _ = planned
    borrowed = coeff.nbytes + r_product.nbytes + jacobians.nbytes
    borrowed += sum(
        item[0].nbytes + (item[4].nbytes if item[4] is not None else 0)
        for item in factors
    )
    operand_copies = 8 * sum(math.prod(item) for item in shapes)
    # The rotation is a fresh contiguous float64 array.  Count every planned
    # result, another copy of each, a separate output and bounded scan buffers.
    workspace = (
        borrowed
        + coeff.size * 8
        + operand_copies
        + 16 * intermediate_elements
        + samples * 8
        + _SCAN_RESERVE
    )
    if workspace > max_bytes:
        return None
    for item in factors:
        if item[4] is not None and not bool(item[4].all()):
            return None
    component = _phase_component(coeff, e_psi)
    if component is None:
        return None
    real_coeff, sign = component
    coefficient_scan = _scan(real_coeff)
    if coefficient_scan is None or coefficient_scan[2] != coeff.size:
        return None
    scans = [coefficient_scan]
    # Inspect complete borrowed matrices, including entries outside pinned
    # slices.  This is deliberately stricter than the ordered selector.
    for array in (*(item[0] for item in factors), r_product, jacobians):
        result = _scan(array)
        if result is None:
            return None
        scans.append(result)
    # Positive exponent sums bound every subset product, even if a later tiny
    # operand would cancel a huge scale.  The complete assignment count bounds
    # every possible reduction.  Four exponent bits reserve overflow headroom.
    upper = (coeff.size - 1).bit_length()
    lower = 0
    for maximum, minimum, _ in scans:
        if maximum:
            upper += max(0, math.frexp(maximum)[1])
        if math.isfinite(minimum):
            # A float64 is an integer multiple of 2**(frexp(x).exp - 53).
            # Products add these powers; reductions preserve their lattice.
            # This also bounds cancellation residuals, not just product sizes.
            lower += min(0, math.frexp(minimum)[1] - 53)
    if upper > 1020 or lower < -1000:
        return None
    rotated = np.array(real_coeff, dtype=np.float64, order="C", copy=True)
    if sign == -1:
        np.negative(rotated, out=rotated)
    selected = [item[0][item[1]] for item in factors]
    arguments = [rotated, list(labels[0])]
    for array, indices in zip((*selected, r_product, jacobians), labels[1:]):
        arguments.extend((array, list(indices)))
    arguments.append([0])
    # No post-hoc global phase projection, and no swallowed backend exceptions.
    return np.einsum(*arguments, optimize=list(path))
