"""The propagator side of a backend contraction: kernel operands and labels.

:func:`kernel_layout` resolves the C and R batches of one integrand call
into einsum operands over integer labels, as the ordered selectors read
them.  :func:`label_classes` merges the labels that structurally diagonal
factors set equal and pins those they fix; :func:`apply_classes` applies
the classes to one operand.  :func:`supports_hold` and
:func:`deferred_values` are the numeric checks of the layout that run
before a contraction.  The rules that admit a contraction are in
:mod:`sft_wick._contraction_admission`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from ._contraction_admission import _float64, _native_dtype, _ordinary_label


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


def deferred_values(deferred):
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


def supports_hold(zero_checks) -> bool:
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
