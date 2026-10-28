"""Shared admission, planning and accuracy contract of the opt-in backends.

The static integrand of a diagram is, per sample ``s``,

    values[s] = sum_a c(a) * prod_k K_k(s, a),

a sum over the component assignments ``a`` of the coefficient ``c`` (the
rotated real part of the coupling array, or for ``"factorized"`` a sum of
monomials ``c(a) = sum_t c_t(a)`` of the original coupling tensors) times the
C and R propagator batches, the scalar R product and the Jacobian.  The
default ``"ordered"`` backend folds the assignments one at a time.
``"einsum"`` and ``"factorized"`` evaluate the same sum as a sequence of
pairwise ``numpy.einsum`` contractions.  An opt-in backend either returns a
result or declines with ``None``; the caller then runs the ordered fold.

Accuracy contract.  Let ``S[s] = sum_a |c(a) prod_k K_k(s, a)|``, ``A`` the
number of component assignments, ``T`` the number of monomials (``T = 1``
for ``"einsum"``) and ``m`` the number of factors in one product.  With
``u = 2**-53``, a returned result satisfies, at every sample,

    |result - exact| <= 1.01 * (m + A*T) * u * S,

where ``exact`` is the sum computed in exact arithmetic from the same
float64 inputs.  The ordered fold satisfies the same bound on every input a
backend admits; the two results generally differ in their last bits.  The
bound has no underflow term because of rule (3) below, and it holds for
``"factorized"`` with the summed coefficient because of rule (4).

Behaviour contract.  A backend only returns a result when (1) every array it
reads is finite, and so is every entry the ordered fold multiplies although
the backend's index structure excludes it (by the zero of a diagonal
``(n, N)`` C table, or as an R factor before the fold reaches the R support
that excludes the assignment), (2) no product or partial sum of those
entries can overflow in any evaluation order, (3) no product of nonzero
inputs can fall below
``2**-1021`` in any evaluation order, so that every underflow error is
smaller than ``u/2`` times the magnitudes it came from (for
``"factorized"`` the prefactor and every rational factor of a monomial are
inputs of their own in (2) and (3), because the ordered preparation
multiplies them one at a time), (4) for
``"factorized"``, the monomials of a sum have no opposite signs at any
assignment, so that ``sum_t |c_t(a)| = |c(a)|``, (5) every coefficient
passes the phase rule of :func:`sft_wick.evaluate._real_or_raise`, (6)
``numpy.geterr()["under"]`` is ``"ignore"`` (``numpy.einsum`` reports no
floating-point errors, so a stricter underflow policy would see different
events), (7) the arrays the call allocates fit ``contraction_max_bytes`` at
the largest total they reach at one time (:func:`einsum_workspace`;
``Contraction.workspace`` and :func:`count_bytes` for ``"factorized"``),
NumPy's iterator buffers included and Python objects excluded, and (8) the
cost model predicts it to be cheaper than the ordered fold.  A
contraction also declines when a summation label is carried by no operand
of a group (:func:`plan_groups`): the ordered fold adds that group's
product once per value of the label.  Under
(1) and (2) neither evaluation order produces inf, NaN or a floating-point
error, so an exception from the ordered fold (phase error, non-isotropic
index-free C) is reproduced by declining.  The output never holds ``-0.0``.

The backends accept the package's own classes and exact float64 or
complex128 arrays only.  Replacing package methods or helpers at run time
(monkeypatching) is unsupported and is not detected; the test suite
replaces :func:`backend_is_cheaper` to run fixtures the cost rule declines.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
import math

import numpy as np

#: Default workspace budget, in bytes, for one coupling contraction (64 MiB).
DEFAULT_CONTRACTION_MAX_BYTES = 64 * 1024 * 1024

#: Largest binary exponent a product or partial sum may reach.  Four bits
#: below the float64 limit leave room for rounding growth.
_MAX_EXPONENT = 1020

#: Smallest binary exponent a product of nonzero inputs may reach: one above
#: the smallest normal exponent.  An underflowing operation then loses at
#: most ``2**-1075 <= (u/2) * 2**-1021`` of the magnitudes it combines.
_MIN_EXPONENT = -1021

#: Largest ``m + A*T`` for which ``1.01 (m + A*T) u`` bounds the rounding.
_MAX_ACCUMULATED = 2**46

#: Coefficients whose rotated imaginary part is nonzero are checked one by
#: one with the ordered projector; above this count the backend declines.
_MAX_PHASE_CHECKS = 4096

QUARTER_TURNS = (1 + 0j, 1j, -1 + 0j, -1j)

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
#: Backend: cost per kernel element of one pass (layout copy, range scans).
_BACKEND_ELEMENT = 0.7
#: Backend: passes over every kernel element (copy, two range scans and the
#: zero-skipping magnitude scan of arrays holding zeros or both signs).
_KERNEL_PASSES = 3.5
#: Backend: passes over a kernel view the ordered fold multiplies beyond the
#: backend's slice (one range scan; see :func:`plan_groups`).
_READ_PASSES = 2.5
#: Backend (einsum): cost per coefficient element (rotation, masks, scan).
_BACKEND_COEFFICIENT = 60.0
#: Backend: fixed cost of one pairwise contraction step.
_BACKEND_STEP = 20000.0
#: Backend: cost of one floating-point operation in a contraction step.
_BACKEND_FLOP = 0.25
#: Backend: fixed cost of one step of the live-assignment count.
_COUNT_STEP = 3000.0
#: Backend (factorized): extra fixed cost of one call.
_FACTORIZED_CALL = 40000.0
#: Backend (factorized): fixed cost per monomial of a sum beyond its steps.
_BACKEND_MONOMIAL = 5000.0
#: Backend (factorized): cost per monomial of a sum and per sample (phase,
#: accumulation).
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


def fixed_labels_ok(fixed) -> bool:
    """Whether fixed indices are a dict of identifier labels to ints."""
    return type(fixed) is dict and all(
        _ordinary_label(k) and type(v) is int for k, v in fixed.items()
    )


def errors_allow_backend() -> bool:
    """Whether the floating-point error policy lets a backend run."""
    return np.geterr()["under"] == "ignore"


_MAGNITUDE_BITS = np.uint64(0x7FFFFFFFFFFFFFFF)
_ONE_BIT = np.uint64(1)

#: Entries a range scan reads at a time.  Its one scratch array holds at
#: most this many 8-byte entries (512 KiB).
_SCAN_CHUNK = 2**16


def _blocks(array):
    """Views of ``array`` of at most ``_SCAN_CHUNK`` entries covering it once.

    Slices of the flattened array when flattening is a view, otherwise runs
    of the first axis; a row longer than ``_SCAN_CHUNK`` is split in turn.
    ``array`` is not empty.  Every block is one-dimensional except the runs
    of the first axis of an array that does not flatten to a view.
    """
    if array.ndim <= 1 or array.flags.c_contiguous:
        flat = array.reshape(-1)
        for start in range(0, flat.size, _SCAN_CHUNK):
            yield flat[start : start + _SCAN_CHUNK]
        return
    row = array.size // array.shape[0]
    if row > _SCAN_CHUNK:
        for item in array:
            yield from _blocks(item)
        return
    step = _SCAN_CHUNK // row
    for start in range(0, array.shape[0], step):
        yield array[start : start + step]


def scan_bytes(elements: int) -> int:
    """Bytes :func:`finite_range` allocates for an array of ``elements`` entries."""
    return 8 * min(elements, _SCAN_CHUNK)


def finite_range(array):
    """``(max|x|, min|x| over nonzero x)``, or ``None`` if an entry is not finite.

    The minimum is ``inf`` when every entry is zero.  ``max`` and ``min``
    propagate NaN and infinities, so two reductions decide finiteness and
    give both values for an array of one strict sign.  Otherwise the
    smallest nonzero magnitude comes from the bit patterns: a finite
    float64 magnitude orders like its uint64 pattern with the sign bit
    cleared, and subtracting one wraps the zero pattern to the largest
    integer, so the minimum skips zeros without a masked reduction.

    The only array this function allocates holds at most :func:`scan_bytes`
    bytes, and no reduction runs over a multidimensional view (NumPy 2.2
    buffers such a reduction).  An array that flattens to a view is reduced
    as one vector, and its bit patterns are formed whole when it has at most
    ``_SCAN_CHUNK`` entries and otherwise in blocks of that many through one
    scratch array.  Any other array is copied into the scratch block by
    block.  Maxima and minima of blocks combine exactly, so the result does
    not depend on the blocks.
    """
    if array.size == 0:
        return 0.0, math.inf
    if array.ndim == 1:
        flat = array
    elif array.ndim == 0 or array.flags.c_contiguous:
        flat = array.reshape(-1)
    else:
        flat = None
    scratch = None
    with np.errstate(all="ignore"):
        if flat is not None:
            high = float(np.max(flat))
            low = float(np.min(flat))
            if not (math.isfinite(high) and math.isfinite(low)):
                return None
        else:
            scratch = np.empty(min(array.size, _SCAN_CHUNK), dtype=np.uint64)
            high, low = -math.inf, math.inf
            for block in _blocks(array):
                copy = scratch.view(np.float64)[: block.size]
                np.copyto(copy.reshape(block.shape), block)
                top = float(np.max(copy))
                bottom = float(np.min(copy))
                if not (math.isfinite(top) and math.isfinite(bottom)):
                    return None
                high = max(high, top)
                low = min(low, bottom)
    maximum = max(abs(high), abs(low))
    if maximum == 0.0:
        return 0.0, math.inf
    if low > 0.0:
        return maximum, low
    if high < 0.0:
        return maximum, -high
    if flat is not None and flat.size <= _SCAN_CHUNK:
        blocks = (flat,)  # one block: its pattern array is the allocation
    else:
        blocks = _blocks(array)
        if scratch is None:
            scratch = np.empty(min(array.size, _SCAN_CHUNK), dtype=np.uint64)
    least = None
    for block in blocks:
        out = None if scratch is None else scratch[: block.size]
        if block.ndim > 1:
            np.copyto(out.reshape(block.shape), block.view(np.uint64))
            source = out
        else:
            source = block.view(np.uint64)
        if low < 0.0:
            bits = np.bitwise_and(source, _MAGNITUDE_BITS, out=out)
            np.subtract(bits, _ONE_BIT, out=bits)
        else:
            # A -0.0 pattern minus one exceeds every finite magnitude.
            bits = np.subtract(source, _ONE_BIT, out=out)
        block_least = bits.min()
        least = block_least if least is None else min(least, block_least)
    smallest = np.array(least + _ONE_BIT, dtype=np.uint64).view(np.float64)
    return maximum, float(smallest)


def exponent_bound(maximum: float) -> int:
    """Smallest ``e >= 0`` with ``maximum < 2**e`` (0 for ``maximum <= 1/2``)."""
    if maximum == 0.0:
        return 0
    return max(0, math.frexp(maximum)[1])


def exponent_floor(smallest: float) -> int:
    """Largest ``e <= 0`` with ``smallest >= 2**e`` (0 for ``inf`` or ``>= 1``)."""
    if not math.isfinite(smallest) or smallest >= 1.0:
        return 0
    return math.frexp(smallest)[1] - 1


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
    if (1j) ** e_psi not in QUARTER_TURNS:
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
        # The count is checked before any index array exists.
        count = np.count_nonzero(other)
        if count > _MAX_PHASE_CHECKS or not _phases_pass(coeff, other, e_psi):
            return None
    return np.multiply(kept, sign)


def _phases_pass(coeff, other, e_psi) -> bool:
    """Whether the ordered projector accepts every coefficient whose rotated
    imaginary part (``other``) is nonzero.  The index array is freed on
    return, before the caller allocates the rotated coefficient."""
    from .evaluate import _real_or_raise

    for position in np.flatnonzero(other):
        try:
            _real_or_raise(coeff.flat[position], e_psi)
        except ValueError:
            return False
    return True


@dataclass(frozen=True)
class KernelLayout:
    """The propagator side of a contraction, as einsum operands.

    Labels are integers: ``0`` is the sample axis and ``1..k`` the summation
    indices.  ``operands`` holds ``(array, labels)`` pairs; an entry whose
    array is ``None`` is an index-free factor computed later from ``deferred``
    (``("isotropic", C)`` or ``("trace", R)``), always with labels ``(0,)``.
    ``deltas`` lists label pairs a structurally diagonal factor sets equal and
    ``pins`` labels it fixes to a component; ``zero`` marks a diagonal factor
    between two different fixed components.  ``skipped_deltas`` and
    ``skipped_pins`` are the subsets that come from the support of R batch
    ``p``, stored as ``(left, right, p)`` and ``(label, value, p)``.  The
    ordered fold multiplies the factors of an assignment a diagonal C table
    excludes by that table's zero; at an assignment an R support excludes it
    stops before the C factors, after multiplying the R batches that precede
    batch ``p``.  ``ranks`` gives, per operand, the last R batch whose
    support the ordered fold checks before it multiplies that operand: ``p``
    for R batch ``p``, ``inf`` (every batch) for C factors and the Jacobian
    and ``-1`` (none) for the scalar R product.  ``supports`` keeps each
    resolved R support with its labels for the ordered cost estimate, and
    ``zero_checks`` the R entries outside their support, which must vanish.
    """

    operands: tuple
    ranks: tuple
    deferred: tuple
    deltas: tuple
    pins: tuple
    skipped_deltas: tuple
    skipped_pins: tuple
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


@dataclass
class _LayoutParts:
    """Mutable accumulator of :func:`kernel_layout`; never escapes it."""

    samples: int
    names: dict
    fixed: dict
    operands: list = field(default_factory=list)
    ranks: list = field(default_factory=list)
    deferred: list = field(default_factory=list)
    deltas: list = field(default_factory=list)
    pins: list = field(default_factory=list)
    skipped_deltas: list = field(default_factory=list)
    skipped_pins: list = field(default_factory=list)
    supports: list = field(default_factory=list)
    zero_checks: list = field(default_factory=list)
    zero: bool = False

    def batch_ok(self, array, ndim) -> bool:
        return (
            _float64(array)
            and array.ndim in ndim
            and array.shape[0] == self.samples
            and bool(array.shape[1])
            and (array.ndim == 2 or array.shape[2] == array.shape[1])
        )

    def pair(self, left, right, dim):
        a = _resolve(left, dim, self.names, self.fixed)
        b = _resolve(right, dim, self.names, self.fixed)
        return None if a is None or b is None else (a, b)

    def add(self, operand, rank) -> None:
        self.operands.append(operand)
        self.ranks.append(rank)

    def diagonal(self, vectors, left, right, rank, skipped=None) -> None:
        """Add a diagonal factor ``delta_ab v[:, a]``.

        ``skipped`` is the index of the R batch whose support this is: the
        ordered fold skips the assignments with ``a != b`` instead of
        multiplying them by zero.
        """
        if left[0] == "sum" and right[0] == "sum":
            self.add((vectors, (0, left[1])), rank)
            if left[1] != right[1]:
                self.deltas.append((left[1], right[1]))
                if skipped is not None:
                    self.skipped_deltas.append((left[1], right[1], skipped))
        elif left[0] == "pin" and right[0] == "pin":
            if left[1] != right[1]:
                self.zero = True
            else:
                self.add((vectors[:, left[1]], (0,)), rank)
        else:
            free, pinned = (left, right) if left[0] == "sum" else (right, left)
            self.add((vectors[:, pinned[1]], (0,)), rank)
            self.pins.append((free[1], pinned[1]))
            if skipped is not None:
                self.skipped_pins.append((free[1], pinned[1], skipped))

    def add_c(self, batch) -> bool:
        if type(batch) is not tuple or len(batch) != 3:
            return False
        array, left, right = batch
        if not self.batch_ok(array, (2, 3)):
            return False
        if left is None and right is None:
            self.add((None, (0,)), math.inf)
            self.deferred.append(("isotropic", array))
            return True
        resolved = self.pair(left, right, array.shape[1])
        if resolved is None:
            return False
        if array.ndim == 3:
            self.add(_select(array, *resolved), math.inf)
        else:
            self.diagonal(array, *resolved, math.inf)
        return True

    def add_r(self, batch, rank) -> bool:
        if type(batch) is not tuple or len(batch) != 4:
            return False
        array, left, right, support = batch
        if not self.batch_ok(array, (3,)):
            return False
        dim = array.shape[1]
        if (
            type(support) is not np.ndarray
            or not _native_dtype(support, np.bool_)
            or support.shape != (dim, dim)
        ):
            return False
        if not (left and right):
            self.add((None, (0,)), rank)
            self.deferred.append(("trace", array))
            return True
        resolved = self.pair(left, right, dim)
        if resolved is None:
            return False
        self.supports.append((support, *resolved))
        on_diagonal = np.diagonal(support)
        if np.count_nonzero(support) == np.count_nonzero(on_diagonal):
            diagonal = np.diagonal(array, axis1=1, axis2=2)
            self.diagonal(diagonal, *resolved, rank, skipped=rank)
            # A diagonal entry outside the support is skipped by the ordered
            # fold, so it must hold zero for the diagonal read to be exact.
            for k in np.flatnonzero(~on_diagonal):
                self.zero_checks.append(diagonal[:, k])
        else:
            self.add(_select(array, *resolved), rank)
            if not support.all():
                self.zero_checks.append((array, ~support))
        return True


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
    parts = _LayoutParts(r_product.size, names, fixed)
    parts.add((r_product, (0,)), -1)
    parts.add((jacobians, (0,)), math.inf)
    if not all(parts.add_c(batch) for batch in c_batches):
        return None
    if not all(parts.add_r(batch, rank) for rank, batch in enumerate(r_batches)):
        return None
    return KernelLayout(
        tuple(parts.operands),
        tuple(parts.ranks),
        tuple(parts.deferred),
        tuple(parts.deltas),
        tuple(parts.pins),
        tuple(parts.skipped_deltas),
        tuple(parts.skipped_pins),
        parts.zero,
        tuple(parts.supports),
        tuple(parts.zero_checks),
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
                    sets[i].isdisjoint(sets[j]),
                    math.prod(dims[label] for label in involved),
                    math.prod(dims[label] for label in kept),
                    i,
                    j,
                )
                if best is None or key < best[0]:
                    best = (key, kept)
        (_, loop, out, i, j), kept = best
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
    (:func:`einsum_buffers`).  Returns ``(None, 0)`` when planning fails.
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


def factorized_overhead(monomials: int, samples: int) -> float:
    """Cost of a factorized call beyond its contraction steps."""
    cost = _FACTORIZED_CALL
    if monomials > 1:
        cost += monomials * (_BACKEND_MONOMIAL + _BACKEND_MONOMIAL_SAMPLE * samples)
    return cost


def minimum_backend_cost(steps: int) -> float:
    """A lower bound of a backend call with ``steps`` contraction steps.

    A product of ``k`` operands takes ``k - 1`` pairwise steps whatever the
    structure merges, so this bound is known before any array is inspected.
    """
    return _BACKEND_CALL + steps * _BACKEND_STEP


def backend_is_cheaper(backend: float, ordered: float) -> bool:
    """The cost rule: run a backend only when it is estimated to be cheaper."""
    return backend < ordered


def restricted_supports(supports):
    """The resolved R supports that exclude an assignment, with their labels.

    Each support is sliced at its fixed components (a boolean view over its
    summation labels); a support that is true everywhere removes no
    assignment and is left out.
    """
    restricted = []
    for support, left, right in supports:
        index = []
        indices = []
        for kind, value in (left, right):
            if kind == "sum":
                index.append(slice(None))
                indices.append(value)
            else:
                index.append(value)
        view = support[tuple(index)]
        if not view.all():
            restricted.append((view, tuple(indices)))
    return tuple(restricted)


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
    return free * int(count[0]), len(steps) * _COUNT_STEP + flops * _BACKEND_FLOP


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
    """Whether every R entry outside its support is zero.

    Each excluded entry is read as a strided column of the batch, so no copy
    of the excluded entries is made; ``count_nonzero`` allocates nothing.
    """
    for check in zero_checks:
        if type(check) is tuple:
            array, outside = check
            for left, right in zip(*np.nonzero(outside)):
                if np.count_nonzero(array[:, left, right]):
                    return False
        elif np.count_nonzero(check):
            return False
    return True


#: Entries per operand of the iterator buffers of ``numpy.einsum``.  It hands
#: its iterator a buffer size of 0, which selects NumPy's compiled default
#: of 8192 entries; ``numpy.setbufsize`` does not change it.
_EINSUM_BUFSIZE = 8192


def iterator_buffers(elements: int) -> int:
    """Bytes NumPy may allocate inside one ufunc or reduction call that
    iterates ``elements`` float64 entries.

    NumPy's iterator can buffer each operand (two inputs and the output) in
    up to ``numpy.getbufsize()`` entries.  Measured with NumPy 2.2 at the
    default buffer size: two buffers (131072 bytes) for a multiplication of
    a strided three-dimensional view and for a maximum over one; with NumPy
    2.4 the same calls allocated less than 64 KiB.  :func:`einsum_buffers`
    counts ``numpy.einsum``, which does not follow ``numpy.getbufsize()``.
    """
    return 24 * min(np.getbufsize(), elements)


def einsum_buffers(elements: int) -> int:
    """Bytes NumPy may allocate inside one ``numpy.einsum`` call whose loop
    space has ``elements`` points.

    Up to ``_EINSUM_BUFSIZE`` entries for each of two inputs and the output,
    whatever ``numpy.getbufsize()`` holds.  Measured with NumPy 2.2: three
    buffers of 8192 entries (196608 bytes) in a step that contracts a
    six-dimensional coefficient with merged labels against a kernel and two
    in a strided outer product, at ``numpy.setbufsize(32)``, 8192 and
    ``2**20`` alike; with NumPy 2.4 the outer product allocated about 64 KiB
    and the other step no buffer.
    """
    return 24 * min(_EINSUM_BUFSIZE, elements)


def layout_bytes(layout) -> int:
    """Bytes of the boolean and index arrays behind the R support checks.

    For every resolved R support of ``n`` entries, :func:`kernel_layout`
    keeps the complement of the support (``n`` bytes) or forms the
    complement of its diagonal and the indices of its excluded diagonal
    entries, :func:`_supports_hold` forms the two index arrays of the
    excluded entries (at most ``16 n`` bytes) and the reductions over the
    support may buffer it (at most ``3 n`` bytes).
    """
    return 20 * sum(support.size for support, _, _ in layout.supports)


@dataclass(frozen=True)
class Contraction:
    """A planned contraction of one or more coefficient groups.

    ``workspace`` is the largest number of bytes the arrays allocated by
    :func:`kernel_layout` and :func:`finish` hold at one time, not counting
    arrays the caller holds while :func:`finish` runs.
    """

    groups: tuple  # ((phase, ((array, labels), ...), steps), ...)
    # ((array or deferred position, sample-last labels, read or None), ...)
    kernels: tuple
    cost: float
    workspace: int


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
    and the planned steps; :func:`factorized_overhead` adds the per-monomial
    work.  The workspace counts, in bytes, :func:`layout_bytes`, then the
    larger of two phases of :func:`finish`: the index-free factors while
    they are computed, and those factors with every sample-last kernel copy
    and either one range scan or the results of every planned step of one
    group (all steps are counted, whether or not their results are alive at
    once) with the iterator buffers of its largest step.
    """
    if layout.zero:
        return None
    classes = label_classes(dims, layout.deltas, layout.pins)
    if classes is None:
        return None
    read_classes = {}
    kernels = []
    shapes = []
    read_sizes = []
    deferred_index = 0
    # Entries of every kernel and of the sample-last copies finish() makes,
    # and of the largest array a range scan reads.
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
    kernel_shapes = tuple(shapes)
    kernel_labels = tuple(labels for _, labels, _ in kernels)
    summed = {value for kind, value in classes.values() if kind == "sum"}
    kernel_carried = set().union(*kernel_labels)
    planned = []
    cost = (
        _BACKEND_CALL
        + _KERNEL_PASSES * elements * _BACKEND_ELEMENT
        + _READ_PASSES * sum(read_sizes) * _BACKEND_ELEMENT
        + samples
    )
    # Bytes of one group's planned steps: every step result and the
    # iterator buffers of its largest step.
    intermediates = 0
    for phase, operands in groups:
        resolved = tuple(apply_classes(a, labels, classes) for a, labels in operands)
        if not summed <= kernel_carried.union(*(indices for _, indices in resolved)):
            return None
        plan, loop = planned_path(
            tuple(array.shape for array, _ in resolved) + kernel_shapes,
            tuple(indices for _, indices in resolved) + kernel_labels,
        )
        if plan is None:
            return None
        steps, flops, step_elements = plan
        cost += len(steps) * _BACKEND_STEP + flops * _BACKEND_FLOP
        intermediates = max(intermediates, 8 * step_elements + einsum_buffers(loop))
        planned.append((phase, resolved, steps))
    if len(groups) > 1:
        # The accumulated result and the previous group's result.
        intermediates += 16 * samples
    deferred_peak = 0
    for kind, array in layout.deferred:
        cost += array.size * (5 if kind == "isotropic" else 1) * _BACKEND_ELEMENT
        # _isotropic_C holds at most three arrays of the batch's size at once,
        # its output among them, and runs ufuncs and reductions; the trace is
        # one einsum, its output counted below.
        if kind == "isotropic":
            held = 24 * array.size + iterator_buffers(array.size)
        else:
            held = einsum_buffers(array.size)
        deferred_peak = max(deferred_peak, held)
    deferred_values = 8 * samples * len(layout.deferred)
    # finish() first computes the index-free factors, then copies and scans
    # the kernels one at a time, then runs the planned steps.
    workspace = layout_bytes(layout) + max(
        deferred_values + deferred_peak,
        deferred_values + 8 * copies + max(scan_bytes(scanned), intermediates),
    )
    return Contraction(tuple(planned), tuple(kernels), cost, workspace), classes


def _range_admits(headroom, ranges) -> bool:
    """Rules (2) and (3) of the module contract for one product."""
    upper = headroom
    lower = 0
    for maximum, smallest in ranges:
        upper += exponent_bound(maximum)
        lower += exponent_floor(smallest)
    return upper <= _MAX_EXPONENT and lower >= _MIN_EXPONENT


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
    if not _supports_hold(layout.zero_checks):
        return None
    deferred = _deferred_values(layout.deferred)
    if deferred is None:
        return None
    kernels = []
    kernel_ranges = []
    for array, labels, read in contraction.kernels:
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
    terms = len(contraction.groups)
    headroom = (assignments - 1).bit_length() + (terms - 1).bit_length()
    for (phase, operands, _), ranges in zip(contraction.groups, coefficient_ranges):
        if any(scanned is None for scanned in ranges):
            return None
        factors = len(operands) + len(kernels) + 1
        if factors + assignments * terms > _MAX_ACCUMULATED:
            return None
        if phase_is_factor:
            magnitude = abs(phase)
            ranges = (*ranges, (magnitude, magnitude if magnitude else math.inf))
        if not _range_admits(headroom, (*kernel_ranges, *ranges)):
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
    # Maps -0.0 to +0.0; the ordered fold starts from +0.0.  Only a negative
    # phase can produce -0.0 here: einsum accumulates into a zeroed output.
    np.add(result, 0.0, out=result)
    return result


def _label_axes(prop_idx):
    """``{label: (axis, size)}`` of the propagator indices, or ``None``."""
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
        or not fixed_labels_ok(fixed_indices)
    ):
        return None
    names = _label_axes(prop_idx)
    if names is None:
        return None
    dims = {axis: dim for axis, dim in names.values()}
    if (
        coeff.shape != tuple(dims.values())
        or coeff.size <= 1
        or type(c_batches) not in (list, tuple)
        or type(r_batches) not in (list, tuple)
        or not _float64(r_product)
    ):
        return None
    samples = r_product.size
    n_c, n_r = len(c_batches), len(r_batches)
    # The ordered estimate with every assignment live bounds it from above;
    # the step count bounds the backend from below.  Small diagrams decline
    # here, before any layout work.
    upper = ordered_cost(coeff.size, coeff.size, coeff.size, n_c, n_r, samples)
    lower = minimum_backend_cost(2 + n_c + n_r) + coeff.size * _BACKEND_COEFFICIENT
    if not backend_is_cheaper(lower, upper):
        return None
    layout = kernel_layout(c_batches, r_batches, r_product, jacobians, names, fixed_indices)
    if layout is None:
        return None
    planned = plan_groups(((1.0, ((coeff, tuple(dims)),)),), layout, dims, samples)
    if planned is None:
        return None
    contraction, classes = planned
    backend = contraction.cost + coeff.size * _BACKEND_COEFFICIENT
    if not backend_is_cheaper(backend, upper):
        return None
    restricted = restricted_supports(layout.supports)
    counting = count_bytes(dims, coeff.shape, restricted)
    if counting is None:
        return None
    if einsum_workspace(coeff.size, contraction, layout, counting) > max_bytes:
        return None
    rotated = rotated_real_coefficient(coeff, e_psi)
    if rotated is None:
        return None
    mask = (rotated != 0.0).astype(np.float64)
    live, count_cost = live_assignments(dims, mask, (), restricted=restricted)
    nonzero = int(np.count_nonzero(mask))
    del mask
    if live is None:
        return None
    estimate = ordered_cost(coeff.size, nonzero, live, layout.n_c, layout.n_r, samples)
    if not backend_is_cheaper(backend + count_cost, estimate):
        return None
    coefficient_range = finite_range(rotated)
    ((_, _, steps),) = contraction.groups
    operand = apply_classes(rotated, tuple(dims), classes)
    contraction = replace(contraction, groups=((1.0, (operand,), steps),))
    return finish(contraction, layout, ((coefficient_range,),), coeff.size)


def einsum_workspace(size, contraction, layout, counting) -> int:
    """Largest number of bytes the arrays of one ``"einsum"`` call hold at once.

    ``size`` is the number of coefficient entries and ``counting`` the
    :func:`count_bytes` of the live-assignment count.  The phase check holds
    a flattened copy of the rotated imaginary part and at most
    ``_MAX_PHASE_CHECKS`` indices; it is over before the rotated coefficient
    (8 bytes per entry) is formed, which NumPy may buffer from its strided
    input.  The rotated coefficient lives to the end of the call.  Before
    :func:`finish` it is joined by the 0/1 mask and, while the mask is
    formed, by a boolean comparison (1 byte per entry), then by the arrays
    of the live count; the mask is freed before the coefficient's range scan
    (:func:`scan_bytes`, at most 8 bytes per entry) and :func:`finish`.
    """
    rotating = 8 * size + max(8 * min(size, _MAX_PHASE_CHECKS), iterator_buffers(size))
    masking = 16 * size + max(size, counting)
    before = layout_bytes(layout) + max(rotating, masking)
    return max(before, 8 * size + contraction.workspace)
