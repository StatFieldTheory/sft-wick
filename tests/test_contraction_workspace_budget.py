"""``contraction_max_bytes`` bounds every array a backend call allocates.

A contraction whose arrays would exceed ``contraction_max_bytes`` declines
rather than allocates, so a call admitted at a budget must not hold more
than that budget in arrays at any time.  For each case the smallest
admitting budget is found by bisection, with the first numeric step after
the budget check (``rotated_real_coefficient`` for ``einsum``,
``live_assignments`` for ``factorized``) replaced by a sentinel; the call
then runs at exactly that budget under ``tracemalloc``.  Its traced peak,
counted from the memory traced before the call (the inputs are excluded,
the output is included), must not exceed the budget plus an allowance.

``tracemalloc`` traces NumPy's array data and iterator buffers, and also
the Python objects a call creates (frames on Python 3.10, the tuples,
lists and dicts of the layout and the plan, ndarray headers of views, the
iterator objects of NumPy calls).  ``contraction_max_bytes`` does not count
those objects.  Their number grows with the einsum operands of a call (the
coupling factors and the kernels of every planned group), and the
allowance is 16 KiB plus 512 bytes per operand (:func:`_allowance`; 48 KiB
at the 64 operands a plan may have).  Over the 422 calls below the traced
peak exceeded the budget by at most 21040 bytes on Python 3.10 with NumPy
2.2 and 20028 bytes on Python 3.14 with NumPy 2.4, both on a four-monomial
factorized sum of 32 operands, whose allowance is 32768 bytes; no call came
closer than 3052 bytes to its allowance.

The cases are generic: diagrams captured from :meth:`Expansion.evaluate` on
a system with distinct rates and signed ``F`` and ``κ³`` (``N`` = 2, 3, 5;
orders 2 and 4; ``diag_C`` off and on, which merges labels; an off-diagonal
component pair, which pins them) at 16, 256 and 8192 samples, kernel-level
layouts the captured diagrams do not reach (full, diagonal, partially
supported and index-free kernels, a pinned delta, an R trace, five and six
summation indices; real and complex coefficients, C-ordered, Fortran-ordered
and strided), and factorized monomials and sums with a full R support, a
partial support sliced at a fixed component, and a partial support over two
summation labels.  Coefficients of six indices at three samples make the
coefficient, its zero mask and the live-assignment count the largest arrays
of a call.  Some cases run at ``numpy.setbufsize(32)``, where
``numpy.einsum`` still buffers 8192 entries per operand and ufuncs buffer
32, and at ``numpy.setbufsize(2**20)``, where ufuncs on strided views
buffer up to that many entries.

With an estimate that left out the coefficient's range-scan bit pattern,
the arrays of the live-assignment count and NumPy's iterator buffers inside
reductions and ``einsum`` steps, 45 of these tests failed on Python 3.14
and 83 on Python 3.10; the largest overrun was 2014364 bytes over a budget
of 4356352 (six indices at ``N`` = 8, Python 3.10).  With ``einsum``
buffers counted at ``numpy.getbufsize()`` entries, 10 and 12 tests failed,
all at ``numpy.setbufsize(32)``, by up to 195172 bytes.  The range scan is
also checked against a whole-array reference on arrays of many scan blocks,
contiguous or not.
"""

import gc
import tracemalloc
from contextlib import contextmanager

import numpy as np
import pytest

import sft_wick._contraction_admission as admission
import sft_wick._dense_contraction as dense
import sft_wick._factorized_coupling as factorized
from sft_wick._dense_contraction import finite_range, try_static_contraction
from sft_wick._factorized_coupling import factorized_state, try_build_factorized
from sft_wick.evaluate import DiagramIntegrand, analyze_spatial
from sft_wick.expressions import Product, Propagator, Rational, Sum, Symbol
from sft_wick.perturbation import DiagramTerm

from .test_contraction_error_bound import (
    _PERMUTATIONS,
    _factorized_term,
    _generic_system,
    _sign_structured,
)

#: Python objects of a call that are not tied to its operands (see the
#: module docstring).
ALLOWANCE_BASE = 16 * 1024
#: Python objects per einsum operand of a call.
ALLOWANCE_PER_OPERAND = 512

#: ``(N, order, diag_C, every k-th diagram)`` of the captured battery.
L1_CASES = [
    (2, 2, False, 1),
    (2, 2, True, 1),
    (3, 2, False, 1),
    (3, 2, True, 1),
    (5, 2, False, 1),
    (2, 4, True, 6),
    (3, 4, True, 6),
]
CAPTURED_SAMPLES = 1024

#: ``numpy.einsum`` buffers this many entries per operand whatever
#: ``numpy.setbufsize`` holds; ufuncs and reductions buffer ``getbufsize()``.
SMALL_BUFFER = 32
LARGE_BUFFER = 2**20


class _Admitted(Exception):
    """Raised in place of the first numeric step after the budget check."""


@pytest.fixture
def admit_all(monkeypatch):
    """Run the backend whatever the cost rule estimates."""
    monkeypatch.setattr(dense, "backend_is_cheaper", lambda backend, ordered: True)


@contextmanager
def _buffer_size(entries):
    """``numpy.setbufsize(entries)`` for the block."""
    previous = np.setbufsize(entries)
    try:
        yield
    finally:
        np.setbufsize(previous)


def _allowance(operands):
    """Bytes of Python objects allowed above the budget for a call with
    ``operands`` einsum operands."""
    return ALLOWANCE_BASE + ALLOWANCE_PER_OPERAND * operands


def _signed(rng, shape, low=0.2, high=1.3):
    """Values of both signs, bounded away from zero, no symmetry."""
    return rng.uniform(low, high, shape) * rng.choice((-1.0, 1.0), shape)


def _resampled(array, samples):
    """``array`` cut or tiled along its sample axis to ``samples`` rows."""
    reps = -(-samples // array.shape[0])
    return np.ascontiguousarray(np.concatenate([array] * reps)[:samples])


def _smallest_budget(call, module, name, monkeypatch):
    """Smallest budget at which ``call`` passes its budget check."""

    def admitted(*args, **kwargs):
        raise _Admitted

    def admits(budget):
        try:
            assert call(budget) is None
        except _Admitted:
            return True
        return False

    with monkeypatch.context() as patch:
        patch.setattr(module, name, admitted)
        high = 2**40
        assert admits(high), "the call declines for a reason other than its budget"
        low = 1
        while low < high:
            middle = (low + high) // 2
            if admits(middle):
                high = middle
            else:
                low = middle + 1
    return low


def _traced_peak(call):
    """``call()`` and the bytes traced at its peak beyond those traced before.

    The garbage of the warm-up call is collected before the measurement and
    the collector is off during it: a collection inside the measured call
    would free objects traced before it and offset the call's own
    allocations (by 8 to 15 KB on the cases below).
    """
    tracing = tracemalloc.is_tracing()
    if not tracing:
        tracemalloc.start()
    enabled = gc.isenabled()
    try:
        call()  # plan caches and lazy imports
        gc.collect()
        gc.disable()
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        result = call()
        return result, tracemalloc.get_traced_memory()[1] - before
    finally:
        if enabled:
            gc.enable()
        if not tracing:
            tracemalloc.stop()


def _measured(cases, module, name, monkeypatch):
    """``(label, budget, peak, operands)`` of every case, each run at its
    smallest budget, which it must not decline."""
    rows = []
    for label, call, operands in cases:
        budget = _smallest_budget(call, module, name, monkeypatch)
        result, peak = _traced_peak(lambda: call(budget))
        assert result is not None, f"{label}: declined at its own budget"
        assert call(budget - 1) is None
        rows.append((label, budget, peak, operands))
    return rows


def _overruns(cases, module, name, monkeypatch):
    """The rows of :func:`_measured` whose peak exceeds the budget plus the
    allowance of the call."""
    return [
        row
        for row in _measured(cases, module, name, monkeypatch)
        if row[2] > row[1] + _allowance(row[3])
    ]


def _einsum_call(coeff, prop_idx, fixed, cb, rp, jac, rb, e_psi):
    def call(budget):
        return try_static_contraction(coeff, prop_idx, fixed, cb, rp, jac, rb, e_psi, budget)

    # The coefficient and every kernel: r_product, jacobians, C and R batches.
    return call, 3 + len(cb) + len(rb)


def _capture():
    """The first static call of every diagram of ``L1_CASES``, at 1024 samples."""
    calls = {}
    original = DiagramIntegrand._static_values
    for n, order, diag, stride in L1_CASES:
        system = _generic_system(n)
        props = system.propagators(t_max=2.0, diag_C=diag, progress=False)
        expansion = system.expand(
            ("phi_a(x)", "phi_b(y)"),
            orders=[order],
            diag_R=diag,
            diag_C=diag,
            progress=False,
        )
        seen = {}

        def record(self, c_batches, r_product, jacobians, *, r_batches=(), where=" (coupling)"):
            if id(self.diagram_term) not in seen:
                seen[id(self.diagram_term)] = (
                    self,
                    [(np.array(c), left, right) for c, left, right in c_batches],
                    np.array(r_product),
                    np.array(jacobians),
                    [(np.array(r), left, right, s) for r, left, right, s in r_batches],
                )
            return original(
                self, c_batches, r_product, jacobians, r_batches=r_batches, where=where
            )

        DiagramIntegrand._static_values = record
        try:
            expansion.evaluate(
                props,
                positions={"x": 0.0, "y": 0.7},
                t_final=1.8,
                component_pair=(0, 1),
                method="qmc_vectorized",
                n_samples=CAPTURED_SAMPLES,
                seed=3,
            )
        finally:
            DiagramIntegrand._static_values = original
        calls[(n, order, diag)] = list(seen.values())[::stride]
    return calls


@pytest.fixture(scope="module")
def captured():
    return _capture()


def _captured_cases(diagrams, samples):
    """``(label, call, operands)`` of every captured diagram at ``samples``."""
    cases = []
    for k, (ig, cb, rp, jac, rb) in enumerate(diagrams):
        call, operands = _einsum_call(
            ig.coupling_array,
            ig.diagram_term.propagator_indices,
            ig.fixed_indices,
            [(_resampled(c, samples), left, right) for c, left, right in cb],
            _resampled(rp, samples),
            _resampled(jac, samples),
            [(_resampled(r, samples), left, right, s) for r, left, right, s in rb],
            ig._e_psi,
        )
        cases.append((f"diagram {k}", call, operands))
    assert cases
    return cases


@pytest.mark.parametrize("samples", [16, 256, 8192])
@pytest.mark.parametrize("key", [case[:3] for case in L1_CASES], ids=str)
def test_einsum_peak_on_captured_diagrams_stays_within_the_budget(
    admit_all, monkeypatch, captured, key, samples
):
    cases = _captured_cases(captured[key], samples)
    assert _overruns(cases, dense, "rotated_real_coefficient", monkeypatch) == []


@pytest.mark.parametrize("key", [case[:3] for case in L1_CASES], ids=str)
def test_einsum_peak_on_captured_diagrams_with_a_small_numpy_buffer(
    admit_all, monkeypatch, captured, key
):
    """At ``numpy.setbufsize(32)`` the planned einsum steps still buffer
    8192 entries per operand."""
    cases = _captured_cases(captured[key], 16)
    with _buffer_size(SMALL_BUFFER):
        assert _overruns(cases, dense, "rotated_real_coefficient", monkeypatch) == []


def _layout(kind, n, samples, rng):
    """Labels and the C and R batches of one generic layout; ``a`` and ``b``
    are fixed to ``n - 1`` and ``0``."""
    s = samples

    def full():
        return _signed(rng, (s, n, n))

    ones = np.ones((n, n), dtype=bool)
    identity = np.eye(n, dtype=bool)
    partial_support = ones.copy()
    partial_support[0, n - 1] = partial_support[n - 1, 1] = partial_support[1, 1] = False
    partial = full()
    partial[:, ~partial_support] = 0.0
    labels = ("i", "j", "k")
    if kind == "full":
        cb = [(full(), "i", "j"), (full(), "k", "a")]
        rb = [(full(), "j", "k", ones)]
    elif kind == "diagonal":
        # A diagonal R support without its (1, 1) entry, a partial support
        # off the diagonal and an index-free C.
        diagonal_support = identity.copy()
        diagonal_support[1, 1] = False
        cb = [
            (_signed(rng, (s, n)), "i", "j"),
            (full(), "k", "a"),
            (np.einsum("s,ab->sab", _signed(rng, s), np.eye(n)), None, None),
        ]
        rb = [
            (full() * diagonal_support, "j", "k", diagonal_support),
            (partial, "i", "b", partial_support),
        ]
    elif kind == "pinned":
        cb = [(_signed(rng, (s, n)), "a", "i"), (full(), "i", "j")]
        rb = [(full(), "j", "k", ones), (full() * identity, "k", "b", identity)]
    elif kind == "trace":
        cb = [(full(), "i", "j"), (full(), "j", "k")]
        rb = [(full(), None, None, ones), (full(), "k", "a", ones)]
    elif kind == "wide":  # five summation indices
        labels = ("i", "j", "k", "l", "m")
        cb = [(full(), "i", "j"), (_signed(rng, (s, n)), "k", "l"), (full(), "m", "a")]
        rb = [(partial, "l", "m", partial_support), (full(), "j", "b", ones)]
    else:  # many, chain: six summation indices on a chain of C batches
        labels = ("i", "j", "k", "l", "m", "n")
        cb = [(full(), left, right) for left, right in zip(labels, labels[1:])]
        cb.append((full(), "n", "a"))
        # chain: full supports, so that no live-assignment count runs.
        if kind == "many":
            rb = [(partial, "i", "k", partial_support), (full(), "m", "b", ones)]
        else:
            rb = [(full(), "i", "k", ones), (full(), "m", "b", ones)]
    return labels, cb, rb


def _coefficient(rng, shape, e_psi, memory="C"):
    """Signed coefficient with zeros; for odd ``e_psi`` complex, with a tiny
    rotated imaginary part on 64 entries so the phase checks run.
    ``memory`` is ``"C"``, ``"F"`` (Fortran order) or ``"strided"`` (every
    other row of a larger array)."""
    real = _signed(rng, shape)
    real.flat[:: 7] = 0.0
    coeff = real
    if e_psi:
        imaginary = np.zeros(shape)
        imaginary.flat[:64] = 1e-13 * real.flat[:64]
        coeff = (real + 1j * imaginary) / (1j**e_psi)
    if memory == "F":
        return np.asfortranarray(coeff)
    if memory == "strided":
        larger = np.zeros((2 * shape[0],) + shape[1:], dtype=coeff.dtype)
        larger[::2] = coeff
        return larger[::2]
    return coeff


def _layout_case(kind, n, samples, memory, e_psi):
    rng = np.random.default_rng(9700 + 10 * n + e_psi)
    labels, cb, rb = _layout(kind, n, samples, rng)
    coeff = _coefficient(rng, (n,) * len(labels), e_psi, memory)
    call, operands = _einsum_call(
        coeff,
        tuple((label, n) for label in labels),
        {"a": n - 1, "b": 0},
        cb,
        _signed(rng, samples),
        _signed(rng, samples),
        rb,
        e_psi,
    )
    return f"{kind} N={n} samples={samples} {memory}", call, operands


LAYOUTS = [
    (kind, n, samples, "C")
    for kind in ("full", "diagonal", "pinned", "trace")
    for n in (3, 6)
    for samples in (16, 512, 9000)
] + [
    ("wide", 3, 16, "C"),
    ("wide", 3, 512, "C"),
    ("wide", 3, 9000, "C"),
    ("wide", 6, 16, "C"),
    ("wide", 6, 64, "C"),
    # Six indices at three samples: the coefficient, its mask and the
    # live-assignment count are the largest arrays of the call.
    ("many", 2, 3, "C"),
    ("many", 5, 3, "F"),
    ("many", 5, 3, "strided"),
    ("many", 8, 3, "C"),
]


@pytest.mark.parametrize("e_psi", [0, 1])
@pytest.mark.parametrize("kind, n, samples, memory", LAYOUTS)
def test_einsum_peak_on_kernel_layouts_stays_within_the_budget(
    admit_all, monkeypatch, kind, n, samples, memory, e_psi
):
    case = _layout_case(kind, n, samples, memory, e_psi)
    assert _overruns([case], dense, "rotated_real_coefficient", monkeypatch) == []


BUFFER_LAYOUTS = [
    ("full", 6, 9000, "C"),
    ("wide", 6, 64, "C"),
    ("many", 5, 3, "F"),
    ("many", 5, 3, "strided"),
    # Rotating a strided coefficient buffers it in a ufunc.
    ("chain", 5, 3, "strided"),
]


@pytest.mark.parametrize("entries", [SMALL_BUFFER, LARGE_BUFFER])
@pytest.mark.parametrize("e_psi", [0, 1])
@pytest.mark.parametrize("kind, n, samples, memory", BUFFER_LAYOUTS)
def test_einsum_peak_at_other_numpy_buffer_sizes(
    admit_all, monkeypatch, kind, n, samples, memory, e_psi, entries
):
    """``numpy.einsum`` buffers 8192 entries per operand at any
    ``numpy.setbufsize``; ufuncs and reductions buffer ``getbufsize()``."""
    case = _layout_case(kind, n, samples, memory, e_psi)
    with _buffer_size(entries):
        assert _overruns([case], dense, "rotated_real_coefficient", monkeypatch) == []


def _summed_r_term(n, kind, scales=(1, Rational(2, 3), Rational(5, 7), 1)):
    """:func:`_factorized_term` with its R between the summation labels
    ``c`` and ``a`` instead of ``c`` and the fixed ``p``."""
    terms = tuple(
        Product((Rational(1) if scale == 1 else scale, Symbol("F", f), Symbol("G", g)))
        for (f, g), scale in zip(_PERMUTATIONS, scales)
    )
    return DiagramTerm(
        (
            Propagator("C", "i", "a", "x", "y"),
            Propagator("C", "j", "b", "x", "y"),
            Propagator("C", "k", "c", "x", "y"),
            Propagator("R", "c", "a", "x", "z"),
        ),
        terms[0] if kind == "monomial" else Sum(terms),
        Rational(3, 7),
        (),
        tuple((label, n) for label in ("i", "j", "k", "a", "b", "c")),
        1,
    )


def _factorized_case(n, kind, diagonal, samples, support):
    """A factorized call; ``support`` is ``"full"``, ``"pinned"`` (the R
    support false at ``(0, 1)`` and ``(n - 1, 1)``, sliced at the fixed
    ``p = 1``) or ``"summed"`` (the same support over the summation labels
    ``c`` and ``a``)."""
    rng = np.random.default_rng(9800 + n)
    if support == "summed":
        term, fixed, right = _summed_r_term(n, kind), {}, "a"
    else:
        term, fixed, right = _factorized_term(n, kind), {"p": 1}, "p"
    values = {"F": -1j * _sign_structured(rng, n), "G": _sign_structured(rng, n)}
    candidate = try_build_factorized(term, analyze_spatial(term), values, fixed, 2**40)
    snapshot = factorized_state(candidate)[1]
    assert snapshot is not None
    allowed = np.ones((n, n), dtype=bool)
    if support != "full":
        allowed[0, 1] = allowed[n - 1, 1] = False
    r = _signed(rng, (samples, n, n))
    r[:, ~allowed] = 0.0
    cb = [
        (_signed(rng, (samples, n, n)), "i", "a"),
        (
            _signed(rng, (samples, n)) if diagonal else _signed(rng, (samples, n, n)),
            "j",
            "b",
        ),
        (_signed(rng, (samples, n, n)), "k", "c"),
    ]
    rb = [(r, "c", right, allowed)]
    rp, jac = _signed(rng, samples), _signed(rng, samples)

    def call(budget):
        return factorized._contract(
            snapshot, dict(fixed), cb, rp, jac, rb, snapshot.e_psi, budget
        )

    kernels = 2 + len(cb) + len(rb)
    operands = sum(len(m.factors) + kernels for m in snapshot.monomials)
    return f"{kind} N={n} samples={samples} {support}", call, operands


@pytest.mark.parametrize("samples", [16, 512, 9000])
@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("kind", ["monomial", "sum"])
@pytest.mark.parametrize("n", [2, 3])
def test_factorized_peak_stays_within_the_budget(
    admit_all, monkeypatch, n, kind, diagonal, samples
):
    case = _factorized_case(n, kind, diagonal, samples, "full")
    assert _overruns([case], factorized, "live_assignments", monkeypatch) == []


@pytest.mark.parametrize("samples", [1, 3, 512])
@pytest.mark.parametrize("support", ["pinned", "summed"])
@pytest.mark.parametrize("kind", ["monomial", "sum"])
@pytest.mark.parametrize("n", [2, 5])
def test_factorized_peak_with_a_partial_r_support_stays_within_the_budget(
    admit_all, monkeypatch, n, kind, support, samples
):
    """A partial support makes the live-assignment count contract the zero
    pattern with the support; at five components and one or three samples
    that count holds more than the planned contraction."""
    case = _factorized_case(n, kind, False, samples, support)
    assert _overruns([case], factorized, "live_assignments", monkeypatch) == []


def _reference_range(array):
    magnitudes = np.abs(array)
    return float(magnitudes.max()), float(magnitudes[magnitudes != 0].min())


@pytest.mark.parametrize(
    "layout", ["contiguous", "fortran", "strided_rows", "long_rows", "one_dimensional"]
)
@pytest.mark.parametrize("smallest", [-(2.0**-1000), 5e-324, -0.3])
def test_range_scan_in_blocks_matches_the_whole_array(monkeypatch, layout, smallest):
    """Arrays of many scan blocks (C-ordered, Fortran-ordered, strided, with
    rows longer than a block, one-dimensional strided), with the smallest
    magnitude in the last block, the largest in a middle one and zeros of
    both signs in the first.  The block is cut to 1000 entries so that small
    arrays span many blocks."""
    monkeypatch.setattr(admission, "_SCAN_CHUNK", 1000, raising=False)
    rng = np.random.default_rng(9900)
    shapes = {
        "contiguous": (40, 50, 7),
        "fortran": (40, 50, 7),
        "strided_rows": (3000, 3, 3),
        "long_rows": (3, 3000, 2),
        "one_dimensional": (30000,),
    }
    base = _signed(rng, shapes[layout], 0.75, 1.3)
    if layout == "fortran":
        array = np.asfortranarray(base)
    elif layout == "strided_rows":
        array = base[:, 1:, ::2]
    elif layout == "long_rows":
        array = base[:, ::-1, :]
    elif layout == "one_dimensional":
        array = base[::3]
    else:
        array = base
    flat_last = tuple(dim - 1 for dim in array.shape)
    array[flat_last] = smallest
    array[(0,) * array.ndim] = 0.0
    array[(0,) * (array.ndim - 1) + (1,)] = -0.0
    array[tuple(dim // 2 for dim in array.shape)] = -2.5
    assert array.size >= 10 * 1000
    assert finite_range(array) == _reference_range(array)
    array[flat_last] = np.nan
    assert finite_range(array) is None


def test_range_scan_allocates_one_block():
    """A scan of 8 MB of signed values with a zero holds one block of 2**16
    entries (512 KiB), not a copy of the whole bit pattern."""
    rng = np.random.default_rng(9910)
    array = _signed(rng, (1000, 1000))
    array[3, 4] = 0.0
    scanned, peak = _traced_peak(lambda: finite_range(array))
    assert scanned == _reference_range(array)
    assert peak <= 8 * 2**16 + _allowance(0)
