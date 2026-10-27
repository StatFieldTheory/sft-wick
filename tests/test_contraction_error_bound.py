"""The error-bound contract of the opt-in contraction backends.

Generic diagrams (N = 3, signed non-symmetric tensors, distinct pins for an
off-diagonal component pair, full, diagonal, partially supported and
index-free kernels) are contracted by ``einsum`` and ``factorized`` and
compared with an exact ``Fraction`` reference that shares no code with the
backends.  Both backends and the ordered fold must lie within the documented
bound ``1.01 (m + A T) u S``, with ``S`` built from the summed coefficient
the ordered fold uses.  Values the ordered fold never reads (behind zero
coefficients or unsupported R entries) must not let a non-finite value
through; overflow and products that can underflow must decline; monomials
of opposite sign must decline; the output holds no ``-0.0``; the planner
keeps a four-tensor monomial inside the default budget; the cost rule
declines a pathological many-monomial sum; realistic L1 diagrams are
admitted.
"""

from dataclasses import replace

import numpy as np
import pytest

import sft_wick as sw
import sft_wick._dense_contraction as dense
import sft_wick.evaluate as ev
from sft_wick._dense_contraction import live_assignments, try_static_contraction
from sft_wick._factorized_coupling import factorized_state, try_build_factorized
from sft_wick.evaluate import analyze_spatial
from sft_wick.expressions import Product, Propagator, Rational, Sum, Symbol
from sft_wick.perturbation import DiagramTerm

from . import _contraction_exact as exact

LIMIT = 64 * 1024**2
N = 3
SAMPLES = 6


@pytest.fixture
def admit_all(monkeypatch):
    """Run the backend on small fixtures, whatever the cost rule estimates."""
    monkeypatch.setattr(
        dense, "backend_is_cheaper", lambda backend, ordered: True, raising=False
    )


def _signed(rng, shape, low=0.2, high=1.3):
    """Values of both signs, bounded away from zero, no symmetry."""
    return rng.uniform(low, high, shape) * rng.choice((-1.0, 1.0), shape)


def _kernels(kind, rng):
    """C and R batches of one generic layout over summation labels i, j, k."""
    c = []
    r = []
    if kind == "full":
        c = [
            (_signed(rng, (SAMPLES, N, N)), "i", "j"),
            (_signed(rng, (SAMPLES, N, N)), "k", "a"),
        ]
        r = [(_signed(rng, (SAMPLES, N, N)), "j", "k", np.ones((N, N), dtype=bool))]
    elif kind == "diagonal":
        identity = np.eye(N, dtype=bool)
        r_diag = _signed(rng, (SAMPLES, N, N)) * identity
        partial = _signed(rng, (SAMPLES, N, N))
        support = np.ones((N, N), dtype=bool)
        support[0, 2] = support[2, 1] = support[1, 1] = False
        partial[:, ~support] = 0.0
        c = [
            (_signed(rng, (SAMPLES, N)), "i", "j"),
            (_signed(rng, (SAMPLES, N, N)), "k", "a"),
            (np.einsum("s,ab->sab", _signed(rng, SAMPLES), np.eye(N)), None, None),
        ]
        r = [(r_diag, "j", "k", identity), (partial, "i", "b", support)]
    elif kind == "pinned_delta":
        identity = np.eye(N, dtype=bool)
        c = [
            (_signed(rng, (SAMPLES, N)), "a", "i"),
            (_signed(rng, (SAMPLES, N, N)), "i", "j"),
        ]
        r = [
            (_signed(rng, (SAMPLES, N, N)), "j", "k", np.ones((N, N), dtype=bool)),
            (_signed(rng, (SAMPLES, N, N)) * identity, "k", "b", identity),
        ]
    else:  # trace of an R with a missing label
        c = [
            (_signed(rng, (SAMPLES, N, N)), "i", "j"),
            (_signed(rng, (SAMPLES, N, N)), "j", "k"),
        ]
        r = [
            (_signed(rng, (SAMPLES, N, N)), None, None, np.ones((N, N), dtype=bool)),
            (_signed(rng, (SAMPLES, N, N)), "k", "a", np.ones((N, N), dtype=bool)),
        ]
    return c, r, _signed(rng, SAMPLES), _signed(rng, SAMPLES)


def _integrand(e_psi, coefficient):
    term = DiagramTerm(
        (Propagator("C", "i", "j", "x", "y"), Propagator("C", "k", "a", "x", "z")),
        Symbol("A", ("i", "j", "k")),
        Rational(1),
        (),
        (("i", N), ("j", N), ("k", N)),
        0,
        n_external_response=e_psi,
    )
    fixed = {"a": 2, "b": 0}
    base = term.build_integrand({"A": np.ones((N, N, N))}, fixed)
    return replace(base, coupling_array=coefficient)


def _coefficient(rng, e_psi, zeros=False):
    """Complex coefficient whose rotated real part is signed; the rotated
    imaginary part is tiny, as the ordered projector tolerates."""
    real = _signed(rng, (N, N, N))
    if zeros:
        real[0, 1, :] = 0.0
        real[2, 2, 2] = 0.0
    rotated = real + 1e-13j * real
    return rotated / (1j**e_psi)


def _ordered(ig, cb, rp, jac, rb):
    return replace(ig, _contraction_backend="ordered")._static_values(
        cb, rp, jac, r_batches=rb
    )


def _einsum(ig, cb, rp, jac, rb, limit=LIMIT):
    return try_static_contraction(
        ig.coupling_array,
        ig.diagram_term.propagator_indices,
        ig.fixed_indices,
        cb,
        rp,
        jac,
        rb,
        ig._e_psi,
        limit,
    )


@pytest.mark.parametrize("kind", ["full", "diagonal", "pinned_delta", "trace"])
@pytest.mark.parametrize("e_psi", [0, 1, 2, 3])
def test_einsum_and_ordered_lie_within_the_bound_of_the_exact_sum(
    admit_all, kind, e_psi
):
    rng = np.random.default_rng(9100 + 7 * e_psi + len(kind))
    ig = _integrand(e_psi, _coefficient(rng, e_psi, zeros=kind == "diagonal"))
    cb, rb, rp, jac = _kernels(kind, rng)
    backend = _einsum(ig, cb, rp, jac, rb)
    assert backend is not None and backend.dtype == np.float64
    ordered = _ordered(ig, cb, rp, jac, rb)
    table = exact.dense_coefficients(ig.coupling_array, e_psi)
    reference, scale = exact.exact_sum(
        table, ig.diagram_term.propagator_indices, ig.fixed_indices, cb, rp, jac, rb
    )
    factors = 1 + len(cb) + len(rb) + 2
    for values in (backend, ordered):
        exact.assert_within_bound(values, reference, scale, factors, N**3)
    assert not np.any(np.signbit(backend) & (backend == 0.0))


@pytest.mark.parametrize("where", ["zero_coefficient", "unsupported_response"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_values_the_ordered_fold_never_reads_decline(admit_all, where, value):
    rng = np.random.default_rng(9200)
    coefficient = _coefficient(rng, 0)
    cb, rb, rp, jac = _kernels("full", rng)
    if where == "zero_coefficient":
        coefficient[:, 1, :] = 0.0
        cb[0][0][:, :, 1] = value  # C(i, j) only at j = 1
    else:
        support = np.ones((N, N), dtype=bool)
        support[1, :] = False
        r_array = rb[0][0]
        r_array[:, 1, :] = 0.0
        rb = [(r_array, "j", "k", support)]
        cb[0][0][:, :, 1] = value  # C(i, j) only where R(j = 1, k) is unsupported
    ig = _integrand(0, coefficient)
    ordered = _ordered(ig, cb, rp, jac, rb)
    assert np.all(np.isfinite(ordered))
    assert _einsum(ig, cb, rp, jac, rb) is None


@pytest.mark.parametrize(
    "scales",
    [(1e308, 1e-308, 1.0, 1.0), (1e200, 1e200, 1e-200, 1e-200), (1e160, 1e160, 1e1, 1e1)],
)
def test_possible_overflow_declines(admit_all, scales):
    rng = np.random.default_rng(9300)
    ig = _integrand(0, np.full((N, N, N), scales[0]))
    cb, rb, rp, jac = _kernels("full", rng)
    cb = [(np.full((SAMPLES, N, N), scales[1]), "i", "j"), cb[1]]
    rp, jac = np.full(SAMPLES, scales[2]), np.full(SAMPLES, scales[3])
    assert _einsum(ig, cb, rp, jac, rb) is None


def _scaled_case(scales, seed=9400):
    """The generic layout with coefficient, C(i, j), pinned C(k, a), rp and
    jac scaled by ``scales``; signs of the coefficient are mixed."""
    rng = np.random.default_rng(seed)
    coefficient = np.full((N, N, N), scales[0]) * rng.choice((-1.0, 1.0), (N, N, N))
    ig = _integrand(0, coefficient)
    cb, rb, rp, jac = _kernels("full", rng)
    cb = [(cb[0][0] * scales[1], "i", "j"), (cb[1][0] * scales[2], "k", "a")]
    return ig, cb, rb, rp * scales[3], jac * scales[4]


@pytest.mark.parametrize(
    "scales",
    [
        (1e-300, 1.0, 1.0, 1.0, 1.0),
        (1e-150, 1e-150, 1.0, 1.0, 1.0),
        (1e-140, 1e-140, 1.0, 1e-10, 1e-10),
        (2.0**-500, 2.0**-480, 2.0**400, 2.0**300, 1.0),
    ],
)
def test_ranges_near_the_underflow_limit_are_admitted_within_the_bound(
    admit_all, scales
):
    """Every product of nonzero inputs stays at or above 2**-1021, and every
    exact result is larger than the bound, so a zero result would fail."""
    ig, cb, rb, rp, jac = _scaled_case(scales)
    backend = _einsum(ig, cb, rp, jac, rb)
    assert backend is not None
    table = exact.dense_coefficients(ig.coupling_array, 0)
    reference, scale = exact.exact_sum(
        table, ig.diagram_term.propagator_indices, ig.fixed_indices, cb, rp, jac, rb
    )
    assert all(exact.bound(size, 6, N**3) < abs(r) for size, r in zip(scale, reference))
    for values in (backend, _ordered(ig, cb, rp, jac, rb)):
        exact.assert_within_bound(values, reference, scale, 6, N**3)


@pytest.mark.parametrize(
    "scales",
    [
        (1e-100, 1e-100, 1.0, 1e-100, 1e-100),
        (np.nextafter(0.0, 1.0), 1.0, 1.0, 1.0, 1.0),
        (1e-160, 1e-160, 1.0, 1e10, 1e10),
        # Normal-range results: 2**-1120 underflows when the coefficient
        # meets C(i, j) first, before the large factors.
        (2.0**-560, 2.0**-560, 2.0**300, 2.0**300, 1.0),
        (2.0**-600, 2.0**-600, 2.0**500, 2.0**500, 1.0),
    ],
)
def test_products_that_can_underflow_decline(admit_all, scales):
    """The ordered fold keeps its value; where the exact result is a normal
    number, that value is accurate to a few units in the last place."""
    ig, cb, rb, rp, jac = _scaled_case(scales)
    assert _einsum(ig, cb, rp, jac, rb) is None
    ordered = _ordered(ig, cb, rp, jac, rb)
    requested = replace(ig, _contraction_backend="einsum")._static_values(
        cb, rp, jac, r_batches=rb
    )
    np.testing.assert_array_equal(requested, ordered)
    table = exact.dense_coefficients(ig.coupling_array, 0)
    reference, _ = exact.exact_sum(
        table, ig.diagram_term.propagator_indices, ig.fixed_indices, cb, rp, jac, rb
    )
    if all(abs(r) > 2.0**-900 for r in reference):
        exact.assert_relative(ordered, reference, 1e-12)


def test_factorized_products_that_can_underflow_decline(admit_all):
    """A monomial of normal-range values whose first contraction step would
    underflow (2**-1120) before large kernels restore the scale."""
    n, samples = 4, 6
    term = DiagramTerm(
        (Propagator("C", "i", "j", "x", "y"), Propagator("C", "a", "b", "x", "z")),
        Product((Rational(1), Symbol("F", ("i", "j")))),
        Rational(1),
        (),
        (("i", n), ("j", n)),
        0,
    )
    rng = np.random.default_rng(9420)
    values = {"F": _signed(rng, (n, n)) * 2.0**-560}
    fixed = {"a": 0, "b": 1}
    candidate = try_build_factorized(term, analyze_spatial(term), values, fixed, LIMIT)
    assert candidate is not None
    cb = [
        (_signed(rng, (samples, n, n)) * 2.0**-560, "i", "j"),
        (_signed(rng, (samples, n, n)) * 2.0**300, "a", "b"),
    ]
    rp, jac = _signed(rng, samples) * 2.0**300, _signed(rng, samples)
    values_out = candidate._static_values(cb, rp, jac, r_batches=[])
    assert factorized_state(candidate)[1] is None, "the call was contracted"
    table = exact.monomial_coefficients(term, values, fixed)
    reference, _ = exact.exact_sum(table, term.propagator_indices, fixed, cb, rp, jac, [])
    exact.assert_relative(values_out, reference, 1e-12)


def test_einsum_output_holds_no_negative_zero(admit_all):
    """A regression guard: ``numpy.einsum`` accumulates into a zeroed output,
    so this path yields +0.0 even without the final normalisation; the
    factorized test below exercises the normalisation itself."""
    rng = np.random.default_rng(9500)
    coefficient = -np.abs(_signed(rng, (N, N, N))).astype(np.complex128)
    ig = _integrand(0, coefficient)
    cb, rb, rp, jac = _kernels("full", rng)
    jac = np.abs(jac)
    jac[::2] = 0.0  # every term at these samples is (-c) * ... * 0
    rp = -np.abs(rp)
    backend = _einsum(ig, cb, rp, jac, rb)
    ordered = _ordered(ig, cb, rp, jac, rb)
    assert backend is not None
    assert np.all(backend[::2] == 0.0) and not np.any(np.signbit(backend[::2]))
    assert not np.any(np.signbit(ordered[::2]))


def test_factorized_negative_phase_output_holds_no_negative_zero(admit_all):
    """A monomial with a negative scalar phase multiplies zero samples by it."""
    rng = np.random.default_rng(9510)
    n = 2
    term = _factorized_term(n, "monomial")
    factors = term.coupling_sum.factors
    term = replace(term, coupling_sum=Product((Rational(-5, 3), *factors[1:])))
    values = {"F": 1j * np.abs(_signed(rng, (n, n, n))), "G": np.abs(_signed(rng, (n, n, n)))}
    fixed = {"p": 0}
    candidate = try_build_factorized(term, analyze_spatial(term), values, fixed, LIMIT)
    baseline = term.build_integrand(values, fixed)
    cb = [(np.abs(_signed(rng, (SAMPLES, n, n))), label, partner) for label, partner in (("i", "a"), ("j", "b"), ("k", "c"))]
    rb = [(np.abs(_signed(rng, (SAMPLES, n, n))), "c", "p", np.ones((n, n), dtype=bool))]
    rp = np.abs(_signed(rng, SAMPLES))
    jac = np.abs(_signed(rng, SAMPLES))
    jac[::2] = 0.0
    fused = candidate._static_values(cb, rp, jac, r_batches=rb)
    assert factorized_state(candidate)[1] is not None, "the runtime call declined"
    assert np.all(fused[::2] == 0.0) and not np.any(np.signbit(fused[::2]))
    assert np.all(fused[1::2] < 0.0)
    assert not np.any(np.signbit(_ordered(baseline, cb, rp, jac, rb)[::2]))


_PERMUTATIONS = (
    (("i", "j", "k"), ("a", "b", "c")),
    (("j", "i", "k"), ("a", "b", "c")),
    (("k", "j", "i"), ("b", "a", "c")),
    (("i", "k", "j"), ("c", "b", "a")),
)


def _factorized_term(n, kind, scales=(1, Rational(2, 3), Rational(5, 7), 1)):
    """A monomial F(i,j,k) G(a,b,c), or a sum of permuted products with the
    rational ``scales``; three C batches, one matrix R."""
    labels = ("i", "j", "k", "a", "b", "c")
    terms = tuple(
        Product((Rational(1) if scale == 1 else scale, Symbol("F", f), Symbol("G", g)))
        for (f, g), scale in zip(_PERMUTATIONS, scales)
    )
    expression = terms[0] if kind == "monomial" else Sum(terms)
    return DiagramTerm(
        (
            Propagator("C", "i", "a", "x", "y"),
            Propagator("C", "j", "b", "x", "y"),
            Propagator("C", "k", "c", "x", "y"),
            Propagator("R", "c", "p", "x", "z"),
        ),
        expression,
        Rational(3, 7),
        (),
        tuple((label, n) for label in labels),
        1,
    )


def _sign_structured(rng, n):
    """Signed, non-symmetric magnitudes whose sign is s(i) s(j) s(k): every
    permuted product of two such tensors has the same sign at an assignment,
    so a sum of them with positive scales has no cancelling monomials."""
    sigma = np.array([1.0, -1.0, -1.0, 1.0, -1.0][:n])
    signs = np.einsum("i,j,k->ijk", sigma, sigma, sigma)
    return signs * rng.uniform(0.2, 1.3, (n, n, n))


@pytest.mark.parametrize("kind", ["monomial", "sum"])
@pytest.mark.parametrize("diagonal", [False, True])
def test_factorized_and_ordered_lie_within_the_bound_of_the_exact_sum(
    admit_all, kind, diagonal
):
    rng = np.random.default_rng(9600 + diagonal)
    n = 2
    term = _factorized_term(n, kind)
    values = {
        "F": -1j * _sign_structured(rng, n),  # build_integrand applies (-i)**1
        "G": _sign_structured(rng, n),
    }
    fixed = {"p": 1}
    baseline = term.build_integrand(values, fixed)
    candidate = try_build_factorized(term, analyze_spatial(term), values, fixed, LIMIT)
    assert candidate is not None
    cb = [
        (_signed(rng, (SAMPLES, n, n)), "i", "a"),
        (
            _signed(rng, (SAMPLES, n)) if diagonal else _signed(rng, (SAMPLES, n, n)),
            "j",
            "b",
        ),
        (_signed(rng, (SAMPLES, n, n)), "k", "c"),
    ]
    rb = [(_signed(rng, (SAMPLES, n, n)), "c", "p", np.ones((n, n), dtype=bool))]
    rp, jac = _signed(rng, SAMPLES), _signed(rng, SAMPLES)
    fused = candidate._static_values(cb, rp, jac, r_batches=rb)
    assert factorized_state(candidate)[1] is not None, "the runtime call declined"
    ordered = _ordered(baseline, cb, rp, jac, rb)
    table = exact.monomial_coefficients(term, values, fixed)
    reference, scale = exact.exact_sum(
        table, term.propagator_indices, fixed, cb, rp, jac, rb
    )
    terms = 1 if kind == "monomial" else 4
    for result in (fused, ordered):
        # Two tensors, one rational, the prefactor and phase, four kernels, rp, jac.
        exact.assert_within_bound(result, reference, scale, 10, n**6, terms)
    assert not np.any(np.signbit(fused) & (fused == 0.0))


@pytest.mark.parametrize("kind", ["compensated", "large_opposite", "permuted"])
def test_factorized_monomials_of_opposite_sign_decline(admit_all, kind):
    """Monomials that cancel would leave an error proportional to their own
    magnitudes; the ordered fold sums them into the coefficient first."""
    rng = np.random.default_rng(9650)
    n = 3
    base = _factorized_term(n, "monomial")
    factors = base.coupling_sum.factors[1:]
    if kind == "compensated":
        values = {"F": -1j * rng.uniform(0.5, 1.5, (n,) * 3), "G": rng.uniform(0.5, 1.5, (n,) * 3)}
        scales = (10**16, 1, -(10**16), 1)
        term = replace(
            base,
            coupling_sum=Sum(tuple(Product((Rational(m), *factors)) for m in scales)),
        )
    elif kind == "large_opposite":
        values = {"F": -1j * _signed(rng, (n,) * 3), "G": _signed(rng, (n,) * 3)}
        scales = (Rational(2**40), Rational(1, 3), Rational(-(2**40)))
        term = replace(
            base, coupling_sum=Sum(tuple(Product((m, *factors)) for m in scales))
        )
    else:
        values = {"F": -1j * _signed(rng, (n,) * 3), "G": _signed(rng, (n,) * 3)}
        scales = (1, Rational(-2, 3), Rational(5, 7), Rational(-1))
        term = _factorized_term(n, "sum", scales)
    fixed = {"p": 1}
    assert try_build_factorized(term, analyze_spatial(term), values, fixed, LIMIT) is None
    requested = term.build_integrand(values, fixed, contraction_backend="factorized")
    ordered = term.build_integrand(values, fixed)
    cb = [
        (_signed(rng, (SAMPLES, n, n)), label, partner)
        for label, partner in (("i", "a"), ("j", "b"), ("k", "c"))
    ]
    rb = [(_signed(rng, (SAMPLES, n, n)), "c", "p", np.ones((n, n), dtype=bool))]
    rp, jac = _signed(rng, SAMPLES), _signed(rng, SAMPLES)
    np.testing.assert_array_equal(
        requested._static_values(cb, rp, jac, r_batches=rb),
        ordered._static_values(cb, rp, jac, r_batches=rb),
    )


@pytest.mark.parametrize("backend", ["einsum", "factorized"])
@pytest.mark.parametrize("entry", ["nonzero", "zero"])
def test_unsupported_diagonal_entry_of_a_diagonal_response(admit_all, backend, entry):
    """An R whose support lies on the diagonal, one diagonal entry outside it.

    The ordered fold skips that entry; a backend reads the diagonal, so it
    must decline when the entry holds a value and stay within the bound when
    it holds zero."""
    rng = np.random.default_rng(9660)
    n = 3
    base = _factorized_term(n, "monomial")
    term = DiagramTerm(
        (*base.propagators[:3], Propagator("R", "c", "a", "x", "z")),
        base.coupling_sum,
        base.rational_prefactor,
        (),
        base.summation_indices,
        1,
    )
    values = {"F": -1j * _signed(rng, (n,) * 3), "G": _signed(rng, (n,) * 3)}
    fixed = {}
    support = np.eye(n, dtype=bool)
    support[1, 1] = False
    response = _signed(rng, (SAMPLES, n, n)) * np.eye(n)
    if entry == "zero":
        response[:, 1, 1] = 0.0
    cb = [
        (_signed(rng, (SAMPLES, n, n)), label, partner)
        for label, partner in (("i", "a"), ("j", "b"), ("k", "c"))
    ]
    rb = [(response, "c", "a", support)]
    rp, jac = _signed(rng, SAMPLES), _signed(rng, SAMPLES)
    ordered = term.build_integrand(values, fixed)
    reference_values = ordered._static_values(cb, rp, jac, r_batches=rb)
    if backend == "einsum":
        result = _einsum(ordered, cb, rp, jac, rb)
    else:
        candidate = try_build_factorized(term, analyze_spatial(term), values, fixed, LIMIT)
        fused = candidate._static_values(cb, rp, jac, r_batches=rb)
        result = fused if factorized_state(candidate)[1] is not None else None
    if entry == "nonzero":
        assert result is None
        return
    assert result is not None
    table = exact.monomial_coefficients(term, values, fixed)
    reference, scale = exact.exact_sum(
        table, term.propagator_indices, fixed, cb, rp, jac, rb
    )
    for values_out in (result, reference_values):
        exact.assert_within_bound(values_out, reference, scale, 10, n**6)


def test_planner_keeps_a_four_tensor_monomial_inside_the_default_budget():
    """F G H K with six matrix R propagators between them, N = 5, 4096
    samples.  Contracting each tensor through its R batches keeps every
    intermediate at N**3 samples; an outer product F (x) G first would not
    fit the default 64 MiB, and the dense coefficient needs 3.9 GB."""
    n, samples = 5, 4096
    labels = [f"l{q}" for q in range(12)]
    pairs = [(0, 3), (1, 4), (2, 5), (6, 9), (7, 10), (8, 11)]
    term = DiagramTerm(
        tuple(Propagator("R", labels[a], labels[b], "x", "y") for a, b in pairs),
        Product(
            (
                Rational(1, 6),
                Symbol("F", tuple(labels[0:3])),
                Symbol("G", tuple(labels[3:6])),
                Symbol("H", tuple(labels[6:9])),
                Symbol("K", tuple(labels[9:12])),
            )
        ),
        Rational(1),
        (),
        tuple((label, n) for label in labels),
        0,
    )
    rng = np.random.default_rng(9670)
    values = {name: _signed(rng, (n, n, n)) for name in "FGHK"}
    candidate = try_build_factorized(term, analyze_spatial(term), values, {}, LIMIT)
    assert candidate is not None
    rb = [
        (_signed(rng, (samples, n, n)), labels[a], labels[b], np.ones((n, n), dtype=bool))
        for a, b in pairs
    ]
    rp, jac = _signed(rng, samples), _signed(rng, samples)
    fused = candidate._static_values([], rp, jac, r_batches=rb)
    assert factorized_state(candidate)[1] is not None, "the call declined"
    # The exact sum factors into two six-label halves.
    from fractions import Fraction
    import itertools

    def half(left, right, rs, s):
        total = magnitude = Fraction(0)
        for a in itertools.product(range(n), repeat=6):
            value = Fraction(float(left[a[:3]])) * Fraction(float(right[a[3:]]))
            for q, r in enumerate(rs):
                value *= Fraction(float(r[s, a[q], a[q + 3]]))
            total += value
            magnitude += abs(value)
        return total, magnitude

    R = [r for r, *_ in rb]
    for s in (0, samples - 1):
        left, left_scale = half(values["F"], values["G"], R[:3], s)
        right, right_scale = half(values["H"], values["K"], R[3:], s)
        outer = Fraction(1, 6) * Fraction(float(rp[s])) * Fraction(float(jac[s]))
        reference = left * right * outer
        scale = left_scale * right_scale * abs(outer)
        exact.assert_within_bound([fused[s]], [reference], [scale], 13, n**12)


def test_live_assignment_count_of_disjoint_supports_is_a_product_of_sums():
    """Six disjoint R supports over twelve labels: the count is the product
    of the six support sizes, computed without visiting N**12 assignments."""
    n = 6
    rng = np.random.default_rng(9680)
    dims = {label: n for label in range(1, 13)}
    supports = []
    expected = 1
    for q in range(6):
        support = rng.random((n, n)) < 0.6
        supports.append((support, ("sum", 2 * q + 1), ("sum", 2 * q + 2)))
        expected *= int(np.count_nonzero(support))
    count, cost = live_assignments(dims, None, supports)
    assert count == expected
    assert cost < 1e6  # one millisecond in the cost model's units


def test_factorized_cost_model_reads_the_coefficient_zero_pattern(admit_all):
    """A structurally sparse coupling: the snapshot counts the nonzero
    coefficients exactly, so the ordered estimate is not a dense one."""
    rng = np.random.default_rng(9690)
    n = 3
    term = _factorized_term(n, "sum", (1, Rational(2, 3), 1, 1))
    term = replace(term, coupling_sum=Sum(term.coupling_sum.terms[:2]))
    F = np.zeros((n, n, n))
    G = np.zeros((n, n, n))
    for a in range(n):
        F[a, a, a] = rng.uniform(0.5, 1.5)
        G[a, (a + 1) % n, (a + 1) % n] = rng.uniform(0.5, 1.5)
    values = {"F": -1j * F, "G": G}
    candidate = try_build_factorized(term, analyze_spatial(term), values, {"p": 0}, LIMIT)
    assert candidate is not None
    snapshot = factorized_state(candidate)[1]
    table = exact.monomial_coefficients(term, values, {"p": 0})
    nonzero = {index for index, (value, _) in table.items() if value}
    assert snapshot.nonzero == len(nonzero) < n**6
    assert {tuple(int(i) for i in index) for index in zip(*np.nonzero(snapshot.mask))} == nonzero


def test_cost_rule_declines_a_pathological_many_monomial_sum(monkeypatch):
    """64 monomials over a 2**6 coefficient: per-monomial work exceeds the
    ordered fold, so the call declines and the coefficient materializes."""
    rng = np.random.default_rng(9700)
    n = 2
    term = _factorized_term(n, "sum")
    many = Sum(tuple(term.coupling_sum.terms) * 16)
    term = replace(term, coupling_sum=many)
    values = {"F": -1j * _sign_structured(rng, n), "G": _sign_structured(rng, n)}
    fixed = {"p": 0}
    candidate = try_build_factorized(term, analyze_spatial(term), values, fixed, LIMIT)
    assert candidate is not None
    decisions = []
    rule = getattr(dense, "backend_is_cheaper", None)
    assert rule is not None

    def record(backend, ordered):
        decisions.append((backend, ordered))
        return rule(backend, ordered)

    monkeypatch.setattr(dense, "backend_is_cheaper", record)
    cb = [
        (_signed(rng, (64, n, n)), "i", "a"),
        (_signed(rng, (64, n, n)), "j", "b"),
        (_signed(rng, (64, n, n)), "k", "c"),
    ]
    rb = [(_signed(rng, (64, n, n)), "c", "p", np.ones((n, n), dtype=bool))]
    rp, jac = _signed(rng, 64), _signed(rng, 64)
    candidate._static_values(cb, rp, jac, r_batches=rb)
    assert decisions and decisions[-1][0] >= decisions[-1][1]
    assert factorized_state(candidate)[1] is None


def _generic_system(n):
    rng = np.random.default_rng(100 + n)
    return sw.System(
        field=sw.FieldSpec("phi", n),
        linear=sw.DiagonalA(gamma=list(0.6 + 0.4 * rng.random(n))),
        vertices=[sw.LocalVertex("F", coupling=rng.normal(size=(n, n, n)) * 0.3)],
        nonlocal_vertices=[
            sw.NonLocalVertex("K", order=3, coupling=rng.normal(size=(n, n, n)) * 0.2)
        ],
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.8, sigma_t=0.5),
                spatial=sw.ExponentialSpatial(sigma_x=1.3),
            )
        ),
    )


def test_realistic_l1_diagrams_are_admitted(monkeypatch):
    import sft_wick._factorized_coupling as factorized

    system = _generic_system(3)
    props = system.propagators(t_max=2.0, diag_C=False, progress=False)
    expansion = system.expand(
        ("phi_a(x)", "phi_b(y)"), orders=[2], diag_R=False, diag_C=False, progress=False
    )
    einsum_calls = []
    contract = ev.try_static_contraction

    def observe_einsum(*args, **kwargs):
        value = contract(*args, **kwargs)
        einsum_calls.append(value is not None)
        return value

    fused_calls = []
    fuse = factorized._contract

    def observe_factorized(*args, **kwargs):
        value = fuse(*args, **kwargs)
        fused_calls.append(value is not None)
        return value

    monkeypatch.setattr(ev, "try_static_contraction", observe_einsum)
    monkeypatch.setattr(factorized, "_contract", observe_factorized)
    options = dict(
        positions={"x": 0.0, "y": 0.7},
        t_final=1.8,
        component_pair=(0, 1),
        method="qmc_vectorized",
        n_samples=4096,
        seed=3,
    )
    ordered = expansion.evaluate(props, **options)
    dense_result = expansion.evaluate(props, contraction_backend="einsum", **options)
    fused_result = expansion.evaluate(props, contraction_backend="factorized", **options)
    assert einsum_calls and sum(einsum_calls) == len(einsum_calls)
    assert fused_calls and any(fused_calls)
    scale = max(abs(d["value"]) for d in ordered.per_diagram)
    for result in (dense_result, fused_result):
        for got, want in zip(result.per_diagram, ordered.per_diagram):
            assert abs(got["value"] - want["value"]) <= 1e-12 * scale
