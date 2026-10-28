"""Admission rules of the opt-in contraction backends, stated once.

The static integrand of a diagram is, per sample ``s``,

    values[s] = sum_a c(a) * prod_k K_k(s, a),

a sum over the component assignments ``a`` of the coefficient ``c`` (the
rotated real part of the coupling array, or for ``"factorized"`` a sum of
monomials ``c(a) = sum_t c_t(a)`` of the original coupling tensors) times the
C and R propagator batches, the scalar R product and the Jacobian.  The
default ``"ordered"`` backend folds the assignments one at a time.
``"einsum"`` (:mod:`sft_wick._dense_contraction`) and ``"factorized"``
(:mod:`sft_wick._factorized_coupling`) evaluate the same sum as a sequence
of pairwise ``numpy.einsum`` contractions.  An opt-in backend either returns
a result or declines with ``None``; the caller then runs the ordered fold.

Accuracy contract.  Let ``S[s] = sum_a |c(a) prod_k K_k(s, a)|``, ``A`` the
number of component assignments, ``T`` the number of monomials (``T = 1``
for ``"einsum"``), ``m`` the number of factors in one product and ``q`` the
number of them that are rationals with no exact float64 value (each the
largest over the products).  For ``"einsum"`` the factors are the
coefficient and the kernels, and ``q = 0``; for ``"factorized"`` they are
the coupling tensors, the rationals and the prefactor of a monomial and the
kernels, and each rational is converted to float64 with one rounding before
it is multiplied.  With ``u = 2**-53``, a returned result satisfies, at
every sample,

    |result - exact| <= 1.01 * (m + q + A*T) * u * S,

where ``exact`` is the sum computed in exact arithmetic from the float64
coefficient (``"einsum"``) or from the entries of the coupling tensors and
the exact rationals (``"factorized"``), and from the float64 kernels.  An
index-free C or R kernel is the float64 value NumPy computes from its table
(``_isotropic_C``, the trace), shared by every path.  The ordered fold
satisfies the same bound on every input a backend admits; the two results
generally differ in their last bits.  The
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
the largest total they reach at one time (:func:`einsum_workspace` and
:func:`factorized_workspace`), NumPy's iterator buffers included and Python
objects excluded, and (8) the cost model predicts it to be cheaper than the
ordered fold.  A contraction also declines when a summation label is
carried by no operand of a group
(:func:`sft_wick._dense_contraction.plan_groups`): the ordered fold adds
that group's product once per value of the label.  Under (1) and (2)
neither evaluation order produces inf, NaN or a floating-point error, so an
exception from the ordered fold (phase error, non-isotropic index-free C)
is reproduced by declining.  The output never holds ``-0.0``.

The backends accept the package's own classes and exact float64 or
complex128 arrays only.  Replacing package methods or helpers at run time
(monkeypatching) is unsupported and is not detected; the test suite
replaces ``sft_wick._dense_contraction.backend_is_cheaper``, through which
both backends apply the cost rule, to run fixtures the rule declines.

This module holds the rules: input types, the finite and exponent range
rules, the phase rule, the cost model and the workspace accounting.  The
planning and execution they guard live in
:mod:`sft_wick._contraction_layout` and :mod:`sft_wick._dense_contraction`.
"""

from __future__ import annotations

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

#: Largest count of coefficient operands and kernels of one product plus
#: ``A*T + 1``.  A monomial adds at most 25 rationals, so ``1.01 (m + q +
#: A*T) u`` still bounds the rounding below this limit.
_MAX_ACCUMULATED = 2**46

#: Coefficients whose rotated imaginary part is nonzero are checked one by
#: one with the ordered projector; above this count the backend declines.
_MAX_PHASE_CHECKS = 4096

QUARTER_TURNS = (1 + 0j, 1j, -1 + 0j, -1j)

#: Most summation labels of one contraction: ``numpy.einsum`` names 52
#: axes, and the sample axis takes one of them.
_MAX_LABELS = 51

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
#: backend's slice (one range scan; see
#: :func:`sft_wick._dense_contraction.plan_groups`).
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


# ---------------------------------------------------------------- inputs


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


def coupling_array_ok(array) -> bool:
    """Whether ``array`` is an exact ndarray of native float64 or complex128."""
    return type(array) is np.ndarray and (
        _native_dtype(array, np.float64) or _native_dtype(array, np.complex128)
    )


def _ordinary_label(value) -> bool:
    # Numeric literals ("1") resolve differently in the ordered selector.
    return type(value) is str and value.isidentifier()


def label_item_ok(item) -> bool:
    """Whether ``item`` is a ``(label, size)`` pair of an identifier label
    and a positive int."""
    return (
        type(item) is tuple
        and len(item) == 2
        and _ordinary_label(item[0])
        and type(item[1]) is int
        and item[1] > 0
    )


def label_axes(items):
    """``{label: (axis, size)}`` of 1 to 51 distinct ``(label, size)`` pairs.

    Axes count from 1 (axis 0 is the sample axis).  ``None`` when ``items``
    is not a tuple of that length, an item fails :func:`label_item_ok`, or a
    label repeats.
    """
    if type(items) is not tuple or not 1 <= len(items) <= _MAX_LABELS:
        return None
    names = {}
    for axis, item in enumerate(items, 1):
        if not label_item_ok(item) or item[0] in names:
            return None
        names[item[0]] = axis, item[1]
    return names


def fixed_labels_ok(fixed) -> bool:
    """Whether fixed indices are a dict of identifier labels to ints."""
    return type(fixed) is dict and all(
        _ordinary_label(k) and type(v) is int for k, v in fixed.items()
    )


def errors_allow_backend() -> bool:
    """Whether the floating-point error policy lets a backend run."""
    return np.geterr()["under"] == "ignore"


def call_inputs_ok(max_bytes, fixed, c_batches, r_batches, r_product) -> bool:
    """The input rules every backend call shares: a positive int budget,
    the underflow policy (rule 6), identifier fixed labels, list or tuple
    batch containers and a float64 scalar R product."""
    return (
        type(max_bytes) is int
        and max_bytes > 0
        and errors_allow_backend()
        and fixed_labels_ok(fixed)
        and type(c_batches) in (list, tuple)
        and type(r_batches) in (list, tuple)
        and _float64(r_product)
    )


# ------------------------------------------------------- range rules (1-3)

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
    scratch = None
    if array.ndim == 1:
        flat = array
    elif array.ndim == 0 or array.flags.c_contiguous:
        flat = array.reshape(-1)
    else:
        flat = None
        scratch = np.empty(min(array.size, _SCAN_CHUNK), dtype=np.uint64)
    with np.errstate(all="ignore"):
        if flat is not None:
            high = float(np.max(flat))
            low = float(np.min(flat))
        else:
            high, low = _blocked_extremes(array, scratch)
    if not (math.isfinite(high) and math.isfinite(low)):
        return None
    maximum = max(abs(high), abs(low))
    if maximum == 0.0:
        return 0.0, math.inf
    if low > 0.0:
        return maximum, low
    if high < 0.0:
        return maximum, -high
    return maximum, _smallest_magnitude(array, flat, low, scratch)


def _blocked_extremes(array, scratch):
    """``(max, min)`` of an array that does not flatten to a view.

    The entries are copied into ``scratch`` block by block.  Returns NaN for
    both at the first block holding an entry that is not finite.
    """
    high, low = -math.inf, math.inf
    for block in _blocks(array):
        copy = scratch.view(np.float64)[: block.size]
        np.copyto(copy.reshape(block.shape), block)
        top = float(np.max(copy))
        bottom = float(np.min(copy))
        if not (math.isfinite(top) and math.isfinite(bottom)):
            return math.nan, math.nan
        high = max(high, top)
        low = min(low, bottom)
    return high, low


def _smallest_magnitude(array, flat, low, scratch) -> float:
    """Smallest nonzero magnitude of finite entries of both signs or zeros."""
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
    return float(smallest)


def scalar_range(magnitude):
    """The :func:`finite_range` of one scalar of magnitude ``magnitude``."""
    return magnitude, magnitude if magnitude else math.inf


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


def range_admits(headroom, ranges) -> bool:
    """Rules (2) and (3) of the module contract for one product."""
    upper = headroom
    lower = 0
    for maximum, smallest in ranges:
        upper += exponent_bound(maximum)
        lower += exponent_floor(smallest)
    return upper <= _MAX_EXPONENT and lower >= _MIN_EXPONENT


def products_admit(groups, coefficient_ranges, kernel_ranges, assignments, phase_is_factor):
    """Rules (1) to (3) and the rounding limit for every group of a call.

    ``groups`` holds ``(phase, operands, steps)``, ``coefficient_ranges``
    the :func:`finite_range` of each group's coefficient operands (``None``
    when one is not finite) and ``kernel_ranges`` those of the kernels.
    Every partial sum adds at most ``log2(assignments) + log2(groups)`` bits.
    With ``phase_is_factor`` the group's phase counts as one more factor.
    """
    terms = len(groups)
    headroom = (assignments - 1).bit_length() + (terms - 1).bit_length()
    for (phase, operands, _), ranges in zip(groups, coefficient_ranges):
        if any(scanned is None for scanned in ranges):
            return False
        factors = len(operands) + len(kernel_ranges) + 1
        if factors + assignments * terms > _MAX_ACCUMULATED:
            return False
        if phase_is_factor:
            ranges = (*ranges, scalar_range(abs(phase)))
        if not range_admits(headroom, (*kernel_ranges, *ranges)):
            return False
    return True


# ------------------------------------------------------- phase rule (5)


def quarter_turn(e_psi) -> bool:
    """Whether ``1j**e_psi`` is an exact quarter turn for an int in [0, 1024]."""
    return type(e_psi) is int and 0 <= e_psi <= 1024 and (1j) ** e_psi in QUARTER_TURNS


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
    if not quarter_turn(e_psi):
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


# --------------------------------------------------------- cost rule (8)


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


def kernel_cost(elements, read_elements, samples) -> float:
    """Fixed cost of a backend call and its passes over the kernels.

    ``elements`` counts the kernel entries the call copies and scans and
    ``read_elements`` the entries of the views it scans beyond them.
    """
    return (
        _BACKEND_CALL
        + _KERNEL_PASSES * elements * _BACKEND_ELEMENT
        + _READ_PASSES * read_elements * _BACKEND_ELEMENT
        + samples
    )


def step_cost(steps, flops) -> float:
    """Cost of a planned contraction of ``steps`` pairwise steps."""
    return steps * _BACKEND_STEP + flops * _BACKEND_FLOP


def count_cost(steps, flops) -> float:
    """Cost of the live-assignment count along its planned steps."""
    return steps * _COUNT_STEP + flops * _BACKEND_FLOP


def deferred_cost(kind, size) -> float:
    """Cost of an index-free factor computed from a batch of ``size`` entries."""
    return size * (5 if kind == "isotropic" else 1) * _BACKEND_ELEMENT


def coefficient_cost(size) -> float:
    """Cost of the ``"einsum"`` coefficient work (rotation, masks, scan)."""
    return size * _BACKEND_COEFFICIENT


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


# -------------------------------------------------- workspace rule (7)

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

    For every resolved R support of ``n`` entries,
    :func:`sft_wick._contraction_layout.kernel_layout` keeps the complement
    of the support (``n`` bytes) or forms the complement of its diagonal and
    the indices of its excluded diagonal entries,
    :func:`sft_wick._contraction_layout.supports_hold` forms the two index
    arrays of the excluded entries (at most ``16 n`` bytes) and the
    reductions over the support may buffer it (at most ``3 n`` bytes).
    """
    return 20 * sum(support.size for support, _, _ in layout.supports)


def step_bytes(step_elements, loop) -> int:
    """Bytes of one group's planned steps: every step result
    (``step_elements`` entries in all) and the iterator buffers of its
    largest step (``loop`` points)."""
    return 8 * step_elements + einsum_buffers(loop)


def deferred_bytes(kind, size) -> int:
    """Bytes held while an index-free factor is computed from a batch of
    ``size`` entries.

    ``_isotropic_C`` holds at most three arrays of the batch's size at once,
    its output among them, and runs ufuncs and reductions; the trace is one
    einsum, its output counted by :func:`finish_workspace`.
    """
    if kind == "isotropic":
        return 24 * size + iterator_buffers(size)
    return einsum_buffers(size)


def finish_workspace(layout, samples, deferred_peak, copies, scanned, intermediates) -> int:
    """Bytes the arrays of :func:`sft_wick._dense_contraction.finish` hold at once.

    :func:`layout_bytes`, then the larger of two phases: the index-free
    factors while they are computed (``deferred_peak``), and those factors
    with every sample-last kernel copy (``copies`` entries) and either one
    range scan of the largest scanned array (``scanned`` entries) or the
    ``intermediates`` bytes of one group's planned steps.
    """
    deferred_values = 8 * samples * len(layout.deferred)
    return layout_bytes(layout) + max(
        deferred_values + deferred_peak,
        deferred_values + 8 * copies + max(scan_bytes(scanned), intermediates),
    )


def einsum_workspace(size, contraction, layout, counting) -> int:
    """Largest number of bytes the arrays of one ``"einsum"`` call hold at once.

    ``size`` is the number of coefficient entries and ``counting`` the
    :func:`sft_wick._dense_contraction.count_bytes` of the live-assignment
    count.  The phase check holds a flattened copy of the rotated imaginary
    part and at most ``_MAX_PHASE_CHECKS`` indices; it is over before the
    rotated coefficient (8 bytes per entry) is formed, which NumPy may buffer
    from its strided input.  The rotated coefficient lives to the end of the
    call.  Before :func:`sft_wick._dense_contraction.finish` it is joined by
    the 0/1 mask and, while the mask is formed, by a boolean comparison (1
    byte per entry), then by the arrays of the live count; the mask is freed
    before the coefficient's range scan (:func:`scan_bytes`, at most 8 bytes
    per entry) and the finish.
    """
    rotating = 8 * size + max(8 * min(size, _MAX_PHASE_CHECKS), iterator_buffers(size))
    masking = 16 * size + max(size, counting)
    before = layout_bytes(layout) + max(rotating, masking)
    return max(before, 8 * size + contraction.workspace)


def factorized_workspace(contraction, layout, counting) -> int:
    """Largest number of bytes the arrays of one ``"factorized"`` call hold at once.

    The snapshot owns the factor copies and the zero mask; a call allocates
    the arrays of the live count (``counting`` bytes) beside those of the
    layout, then the arrays of the finish (``contraction.workspace``).
    """
    return max(layout_bytes(layout) + counting, contraction.workspace)
