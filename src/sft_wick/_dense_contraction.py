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
reads is finite, (2) no product or partial sum can overflow in any
evaluation order, (3) no product of nonzero inputs can fall below
``2**-1021`` in any evaluation order, so that every underflow error is
smaller than ``u/2`` times the magnitudes it came from, (4) for
``"factorized"``, the monomials of a sum have no opposite signs at any
assignment, so that ``sum_t |c_t(a)| = |c(a)|``, (5) every coefficient
passes the phase rule of :func:`sft_wick.evaluate._real_or_raise`, (6)
``numpy.geterr()["under"]`` is ``"ignore"`` (``numpy.einsum`` reports no
floating-point errors, so a stricter underflow policy would see different
events), (7) the planned array workspace fits ``contraction_max_bytes`` and
(8) the cost model predicts it to be cheaper than the ordered fold.  Under
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


def finite_range(array):
    """``(max|x|, min|x| over nonzero x)``, or ``None`` if an entry is not finite.

    The minimum is ``inf`` when every entry is zero.  ``max`` and ``min``
    propagate NaN and infinities, so two reductions decide finiteness and
    give both values for an array of one strict sign.  Otherwise the
    smallest nonzero magnitude comes from the bit patterns: a finite
    float64 magnitude orders like its uint64 pattern with the sign bit
    cleared, and subtracting one wraps the zero pattern to the largest
    integer, so the minimum skips zeros without a masked reduction.
    """
    if array.size == 0:
        return 0.0, math.inf
    with np.errstate(all="ignore"):
        high = float(np.max(array))
        low = float(np.min(array))
    if not (math.isfinite(high) and math.isfinite(low)):
        return None
    maximum = max(abs(high), abs(low))
    if maximum == 0.0:
        return 0.0, math.inf
    if low > 0.0:
        return maximum, low
    if high < 0.0:
        return maximum, -high
    bits = array.view(np.uint64)
    if low < 0.0:
        bits = np.bitwise_and(bits, _MAGNITUDE_BITS)
        np.subtract(bits, _ONE_BIT, out=bits)
    else:
        # A -0.0 pattern minus one exceeds every finite magnitude.
        bits = np.subtract(bits, _ONE_BIT)
    smallest = np.array(bits.min() + _ONE_BIT, dtype=np.uint64).view(np.float64)
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


@dataclass
class _LayoutParts:
    """Mutable accumulator of :func:`kernel_layout`; never escapes it."""

    samples: int
    names: dict
    fixed: dict
    operands: list = field(default_factory=list)
    deferred: list = field(default_factory=list)
    deltas: list = field(default_factory=list)
    pins: list = field(default_factory=list)
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

    def diagonal(self, vectors, left, right) -> None:
        """Add a diagonal factor ``delta_ab v[:, a]``."""
        if left[0] == "sum" and right[0] == "sum":
            self.operands.append((vectors, (0, left[1])))
            if left[1] != right[1]:
                self.deltas.append((left[1], right[1]))
        elif left[0] == "pin" and right[0] == "pin":
            if left[1] != right[1]:
                self.zero = True
            else:
                self.operands.append((vectors[:, left[1]], (0,)))
        else:
            free, pinned = (left, right) if left[0] == "sum" else (right, left)
            self.operands.append((vectors[:, pinned[1]], (0,)))
            self.pins.append((free[1], pinned[1]))

    def add_c(self, batch) -> bool:
        if type(batch) is not tuple or len(batch) != 3:
            return False
        array, left, right = batch
        if not self.batch_ok(array, (2, 3)):
            return False
        if left is None and right is None:
            self.operands.append((None, (0,)))
            self.deferred.append(("isotropic", array))
            return True
        resolved = self.pair(left, right, array.shape[1])
        if resolved is None:
            return False
        if array.ndim == 3:
            self.operands.append(_select(array, *resolved))
        else:
            self.diagonal(array, *resolved)
        return True

    def add_r(self, batch) -> bool:
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
            self.operands.append((None, (0,)))
            self.deferred.append(("trace", array))
            return True
        resolved = self.pair(left, right, dim)
        if resolved is None:
            return False
        self.supports.append((support, *resolved))
        on_diagonal = np.diagonal(support)
        if np.count_nonzero(support) == np.count_nonzero(on_diagonal):
            diagonal = np.diagonal(array, axis1=1, axis2=2)
            self.diagonal(diagonal, *resolved)
            # A diagonal entry outside the support is skipped by the ordered
            # fold, so it must hold zero for the diagonal read to be exact.
            for k in np.flatnonzero(~on_diagonal):
                self.zero_checks.append(diagonal[:, k])
        else:
            self.operands.append(_select(array, *resolved))
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
    parts.operands.extend(((r_product, (0,)), (jacobians, (0,))))
    if not all(parts.add_c(batch) for batch in c_batches):
        return None
    if not all(parts.add_r(batch) for batch in r_batches):
        return None
    return KernelLayout(
        tuple(parts.operands),
        tuple(parts.deferred),
        tuple(parts.deltas),
        tuple(parts.pins),
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


def live_assignments(dims, coefficient_mask, supports):
    """Number of assignments the ordered fold multiplies out, and its cost.

    ``dims`` maps the summation labels to sizes, ``coefficient_mask`` is a
    0/1 float array over all of them in order (or ``None`` for a dense
    coefficient) and ``supports`` the resolved R supports.  The count is
    exact (float64 holds integers below 2**53).  It is contracted along a
    planned pairwise path, so disjoint supports cost the sum, not the
    product, of their sizes.  Returns ``(count, estimated cost)``.
    """
    operands = []
    used = set()
    for support, left, right in supports:
        index = []
        indices = []
        for kind, value in (left, right):
            if kind == "sum":
                index.append(slice(None))
                indices.append(value)
            else:
                index.append(value)
        restricted = support[tuple(index)]
        if restricted.all():
            continue  # a full support removes no assignment
        operands.append((restricted.astype(np.float64), tuple(indices)))
        used.update(indices)
    if coefficient_mask is not None:
        if not operands:
            return int(np.count_nonzero(coefficient_mask)), 0.0
        operands.append((coefficient_mask, tuple(dims)))
        used.update(dims)
    free = math.prod(size for label, size in dims.items() if label not in used)
    if not operands:
        return free, 0.0
    # A one-element sample axis gives the planner its required output label.
    operands.append((np.ones(1), (0,)))
    plan = plan_contraction(
        tuple(array.shape for array, _ in operands),
        tuple(labels for _, labels in operands),
    )
    if plan is None:
        return None, math.inf
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
    structurally or a plan fails.  The cost covers the kernel passes and the
    planned steps; :func:`factorized_overhead` adds the per-monomial work.
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
    cost = _BACKEND_CALL + _KERNEL_PASSES * elements * _BACKEND_ELEMENT + samples
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
        workspace = max(workspace, step_elements)
        planned.append((phase, resolved, steps))
    for kind, array in layout.deferred:
        cost += array.size * (5 if kind == "isotropic" else 1) * _BACKEND_ELEMENT
        workspace += 3 * array.size if kind == "isotropic" else samples
    for check in layout.zero_checks:
        if type(check) is tuple:
            workspace += samples * int(np.count_nonzero(check[1]))
    # Sample-last kernel copies, the bit-pattern temporary of one range scan,
    # the accumulator and one group result.
    largest = max(math.prod(shape) for shape in kernel_shapes)
    workspace += elements + largest + samples * (2 if len(groups) > 1 else 1)
    return Contraction(tuple(planned), tuple(kernels), cost, 8 * workspace), classes


def _range_admits(headroom, ranges) -> bool:
    """Rules (2) and (3) of the module contract for one product."""
    upper = headroom
    lower = 0
    for maximum, smallest in ranges:
        upper += exponent_bound(maximum)
        lower += exponent_floor(smallest)
    return upper <= _MAX_EXPONENT and lower >= _MIN_EXPONENT


def finish(contraction, layout, coefficient_ranges, assignments):
    """Numeric admission and execution of a planned contraction.

    ``coefficient_ranges`` holds, per group, the :func:`finite_range` of each
    coefficient operand (``None`` when one is not finite).  Returns the
    sample vector or ``None``.
    """
    if not _supports_hold(layout.zero_checks):
        return None
    deferred = _deferred_values(layout.deferred)
    if deferred is None:
        return None
    kernels = []
    kernel_ranges = []
    for array, labels in contraction.kernels:
        if type(array) is int:
            array = deferred[array]
        elif array.ndim > 1 and labels[-1] == 0:
            array = np.ascontiguousarray(np.moveaxis(array, 0, -1))
        scanned = finite_range(array)
        if scanned is None:
            return None
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
        magnitude = abs(phase)
        scale = (magnitude, magnitude if magnitude else math.inf)
        if not _range_admits(headroom, (*kernel_ranges, *ranges, scale)):
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
    # The rotated coefficient and its zero mask.
    if contraction.workspace + 16 * coeff.size > max_bytes:
        return None
    backend = contraction.cost + coeff.size * _BACKEND_COEFFICIENT
    if not backend_is_cheaper(backend, upper):
        return None
    rotated = rotated_real_coefficient(coeff, e_psi)
    if rotated is None:
        return None
    mask = (rotated != 0.0).astype(np.float64)
    live, count_cost = live_assignments(dims, mask, layout.supports)
    if live is None:
        return None
    estimate = ordered_cost(
        coeff.size,
        int(np.count_nonzero(mask)),
        live,
        layout.n_c,
        layout.n_r,
        samples,
    )
    if not backend_is_cheaper(backend + count_cost, estimate):
        return None
    ((_, _, steps),) = contraction.groups
    operand = apply_classes(rotated, tuple(dims), classes)
    contraction = replace(contraction, groups=((1.0, (operand,), steps),))
    return finish(contraction, layout, ((finite_range(rotated),),), coeff.size)
