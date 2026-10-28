"""``contraction_max_bytes`` bounds every array a backend call allocates.

A contraction whose arrays would exceed ``contraction_max_bytes`` declines
rather than allocates, so a call admitted at a budget must not hold more
than that budget in arrays at any time.  For each case the smallest
admitting budget is found by bisection, with the first numeric step after
the budget check (``rotated_real_coefficient`` for ``einsum``,
``live_assignments`` for ``factorized``) replaced by a sentinel; the call
then runs at exactly that budget under ``tracemalloc``.  Its traced peak,
counted from the memory traced before the call (the inputs are excluded,
the output is included), must not exceed the budget plus ``ALLOWANCE``.

``tracemalloc`` traces NumPy's array data and iterator buffers, and also
the Python objects a call creates (frames on Python 3.10, the tuples,
lists and dicts of the layout and the plan, ndarray headers of views, the
iterator objects of NumPy calls).  ``contraction_max_bytes`` does not count
those objects; their number grows with the operands and steps of a call,
not with its arrays.  ``ALLOWANCE`` covers them: over the 274 ``einsum``
and 24 ``factorized`` calls below the traced peak exceeded the budget by at
most 7216 bytes on Python 3.10 with NumPy 2.2 and 2664 bytes on Python 3.14
with NumPy 2.4.

The cases are generic: diagrams captured from :meth:`Expansion.evaluate` on
a system with distinct rates and signed ``F`` and ``κ³`` (``N`` = 2, 3, 5;
orders 2 and 4; ``diag_C`` off and on, which merges labels; an off-diagonal
component pair, which pins them) at 16, 256 and 8192 samples, and
kernel-level layouts the captured diagrams do not reach (full, diagonal,
partially supported and index-free kernels, a pinned delta, an R trace, a
five-index coefficient; real and complex coefficients) and factorized sums.
With an estimate that left out the coefficient's range-scan bit pattern,
the arrays of the live-assignment count and NumPy's iterator buffers inside
reductions and ``einsum`` steps, the peak exceeded the budget by up to
197600 bytes (Python 3.10) and 123864 bytes (Python 3.14).  The range scan
is also checked against a whole-array reference on arrays of many scan
blocks, contiguous or not.
"""

import tracemalloc

import numpy as np
import pytest

import sft_wick._dense_contraction as dense
import sft_wick._factorized_coupling as factorized
from sft_wick._dense_contraction import finite_range, try_static_contraction
from sft_wick._factorized_coupling import factorized_state, try_build_factorized
from sft_wick.evaluate import DiagramIntegrand, analyze_spatial

from .test_contraction_error_bound import (
    _factorized_term,
    _generic_system,
    _sign_structured,
)

#: Python objects a call creates beyond its arrays (see the module docstring).
ALLOWANCE = 16 * 1024

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


class _Admitted(Exception):
    """Raised in place of the first numeric step after the budget check."""


@pytest.fixture
def admit_all(monkeypatch):
    """Run the backend whatever the cost rule estimates."""
    monkeypatch.setattr(dense, "backend_is_cheaper", lambda backend, ordered: True)


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
    """``call()`` and the bytes traced at its peak beyond those traced before."""
    tracing = tracemalloc.is_tracing()
    if not tracing:
        tracemalloc.start()
    try:
        call()  # plan caches and lazy imports
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        result = call()
        return result, tracemalloc.get_traced_memory()[1] - before
    finally:
        if not tracing:
            tracemalloc.stop()


def _overruns(cases, module, name, monkeypatch):
    """``(label, budget, peak)`` of every case whose peak exceeds its budget
    plus ``ALLOWANCE``; every case must run at its smallest budget."""
    overruns = []
    for label, call in cases:
        budget = _smallest_budget(call, module, name, monkeypatch)
        result, peak = _traced_peak(lambda: call(budget))
        assert result is not None, f"{label}: declined at its own budget"
        assert call(budget - 1) is None
        if peak > budget + ALLOWANCE:
            overruns.append((label, budget, peak))
    return overruns


def _einsum_call(coeff, prop_idx, fixed, cb, rp, jac, rb, e_psi):
    def call(budget):
        return try_static_contraction(coeff, prop_idx, fixed, cb, rp, jac, rb, e_psi, budget)

    return call


@pytest.fixture(scope="module")
def captured():
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


@pytest.mark.parametrize("samples", [16, 256, 8192])
@pytest.mark.parametrize("key", [case[:3] for case in L1_CASES], ids=str)
def test_einsum_peak_on_captured_diagrams_stays_within_the_budget(
    admit_all, monkeypatch, captured, key, samples
):
    cases = []
    for k, (ig, cb, rp, jac, rb) in enumerate(captured[key]):
        call = _einsum_call(
            ig.coupling_array,
            ig.diagram_term.propagator_indices,
            ig.fixed_indices,
            [(_resampled(c, samples), left, right) for c, left, right in cb],
            _resampled(rp, samples),
            _resampled(jac, samples),
            [(_resampled(r, samples), left, right, s) for r, left, right, s in rb],
            ig._e_psi,
        )
        cases.append((f"diagram {k}", call))
    assert cases
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
    else:  # wide: five summation indices
        labels = ("i", "j", "k", "l", "m")
        cb = [(full(), "i", "j"), (_signed(rng, (s, n)), "k", "l"), (full(), "m", "a")]
        rb = [(partial, "l", "m", partial_support), (full(), "j", "b", ones)]
    return labels, cb, rb


def _coefficient(rng, shape, e_psi):
    """Signed coefficient with zeros; for odd ``e_psi`` complex, with a tiny
    rotated imaginary part on 64 entries so the phase checks run."""
    real = _signed(rng, shape)
    real.flat[:: 7] = 0.0
    if e_psi == 0:
        return real
    imaginary = np.zeros(shape)
    imaginary.flat[:64] = 1e-13 * real.flat[:64]
    return (real + 1j * imaginary) / (1j**e_psi)


LAYOUTS = [
    (kind, n, samples)
    for kind in ("full", "diagonal", "pinned", "trace")
    for n in (3, 6)
    for samples in (16, 512, 9000)
] + [("wide", 3, 16), ("wide", 3, 512), ("wide", 3, 9000), ("wide", 6, 16), ("wide", 6, 64)]


@pytest.mark.parametrize("e_psi", [0, 1])
@pytest.mark.parametrize("kind, n, samples", LAYOUTS)
def test_einsum_peak_on_kernel_layouts_stays_within_the_budget(
    admit_all, monkeypatch, kind, n, samples, e_psi
):
    rng = np.random.default_rng(9700 + 10 * n + e_psi)
    labels, cb, rb = _layout(kind, n, samples, rng)
    coeff = _coefficient(rng, (n,) * len(labels), e_psi)
    call = _einsum_call(
        coeff,
        tuple((label, n) for label in labels),
        {"a": n - 1, "b": 0},
        cb,
        _signed(rng, samples),
        _signed(rng, samples),
        rb,
        e_psi,
    )
    assert _overruns([(kind, call)], dense, "rotated_real_coefficient", monkeypatch) == []


@pytest.mark.parametrize("samples", [16, 512, 9000])
@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("kind", ["monomial", "sum"])
@pytest.mark.parametrize("n", [2, 3])
def test_factorized_peak_stays_within_the_budget(
    admit_all, monkeypatch, n, kind, diagonal, samples
):
    rng = np.random.default_rng(9800 + n)
    term = _factorized_term(n, kind)
    values = {"F": -1j * _sign_structured(rng, n), "G": _sign_structured(rng, n)}
    fixed = {"p": 1}
    candidate = try_build_factorized(term, analyze_spatial(term), values, fixed, 2**40)
    snapshot = factorized_state(candidate)[1]
    assert snapshot is not None
    cb = [
        (_signed(rng, (samples, n, n)), "i", "a"),
        (
            _signed(rng, (samples, n)) if diagonal else _signed(rng, (samples, n, n)),
            "j",
            "b",
        ),
        (_signed(rng, (samples, n, n)), "k", "c"),
    ]
    rb = [(_signed(rng, (samples, n, n)), "c", "p", np.ones((n, n), dtype=bool))]
    rp, jac = _signed(rng, samples), _signed(rng, samples)

    def call(budget):
        return factorized._contract(
            snapshot, dict(fixed), cb, rp, jac, rb, snapshot.e_psi, budget
        )

    assert _overruns([(kind, call)], factorized, "live_assignments", monkeypatch) == []


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
    monkeypatch.setattr(dense, "_SCAN_CHUNK", 1000, raising=False)
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
    assert peak <= 8 * 2**16 + ALLOWANCE
