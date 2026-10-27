"""The error-bound contract of the opt-in contraction backends.

Generic diagrams (N = 3, signed non-symmetric tensors, distinct pins for an
off-diagonal component pair, full, diagonal, partially supported and
index-free kernels) are contracted by ``einsum`` and ``factorized`` and
compared with an exact ``Fraction`` reference that shares no code with the
backends.  Both backends and the ordered fold must lie within the documented
bound ``1.01 (m + A T) u S + m A T 2**-1074 P``.  Values the ordered fold
never reads (behind zero coefficients or unsupported R entries) must not let
a non-finite value through; overflow must decline; the output holds no
``-0.0``; the cost rule declines a pathological many-monomial sum; realistic
L1 diagrams are admitted.
"""

from dataclasses import replace

import numpy as np
import pytest

import sft_wick as sw
import sft_wick._dense_contraction as dense
import sft_wick.evaluate as ev
from sft_wick._dense_contraction import try_static_contraction
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


def _maxima_product(*arrays):
    product = 1.0
    for array in arrays:
        product *= max(1.0, float(np.max(np.abs(array))))
    return product


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
    product = _maxima_product(
        ig.coupling_array, rp, jac, *(a for a, *_ in cb), *(a for a, *_ in rb)
    )
    for values in (backend, ordered):
        exact.assert_within_bound(
            values, reference, scale, factors, N**3, maxima_product=product
        )
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


@pytest.mark.parametrize(
    "scales",
    [
        (1e-300, 1.0, 1.0, 1.0),
        (1e-100, 1e-100, 1e-100, 1e-100),
        (np.nextafter(0.0, 1.0), 1.0, 1.0, 1.0),
        (1e-160, 1e-160, 1e10, 1e10),
    ],
)
def test_underflow_ranges_are_admitted_within_the_bound(admit_all, scales):
    rng = np.random.default_rng(9400)
    coefficient = np.full((N, N, N), scales[0]) * rng.choice((-1.0, 1.0), (N, N, N))
    ig = _integrand(0, coefficient)
    cb, rb, rp, jac = _kernels("full", rng)
    cb = [(cb[0][0] * scales[1], "i", "j"), cb[1]]
    rp, jac = rp * scales[2], jac * scales[3]
    backend = _einsum(ig, cb, rp, jac, rb)
    assert backend is not None
    table = exact.dense_coefficients(ig.coupling_array, 0)
    reference, scale = exact.exact_sum(
        table, ig.diagram_term.propagator_indices, ig.fixed_indices, cb, rp, jac, rb
    )
    product = _maxima_product(coefficient, rp, jac, *(a for a, *_ in cb), rb[0][0])
    for values in (backend, _ordered(ig, cb, rp, jac, rb)):
        exact.assert_within_bound(values, reference, scale, 6, N**3, 1, product)


def test_einsum_output_holds_no_negative_zero(admit_all):
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


def _factorized_term(n, kind):
    """Signed sum of permuted F(i,j,k) G(a,b,c) products, one diagonal C."""
    labels = ("i", "j", "k", "a", "b", "c")
    perms = [
        (("i", "j", "k"), ("a", "b", "c"), Rational(1)),
        (("j", "i", "k"), ("a", "b", "c"), Rational(-2, 3)),
        (("k", "j", "i"), ("b", "a", "c"), Rational(5, 7)),
        (("i", "k", "j"), ("c", "b", "a"), Rational(-1)),
    ]
    terms = tuple(
        Product((scale, Symbol("F", f), Symbol("G", g))) for f, g, scale in perms
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


@pytest.mark.parametrize("kind", ["monomial", "signed_sum"])
@pytest.mark.parametrize("diagonal", [False, True])
def test_factorized_and_ordered_lie_within_the_bound_of_the_exact_sum(
    admit_all, kind, diagonal
):
    rng = np.random.default_rng(9600 + diagonal)
    n = 2
    term = _factorized_term(n, kind)
    values = {
        "F": -1j * _signed(rng, (n, n, n)),  # build_integrand applies (-i)**1
        "G": _signed(rng, (n, n, n)),
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
    product = _maxima_product(
        values["F"], values["G"], rp, jac, *(a for a, *_ in cb), rb[0][0]
    )
    for result in (fused, ordered):
        # Two tensors, one rational, the prefactor and phase, four kernels, rp, jac.
        exact.assert_within_bound(
            result, reference, scale, 10, n**6, terms, maxima_product=product
        )
    assert not np.any(np.signbit(fused) & (fused == 0.0))


def test_cost_rule_declines_a_pathological_many_monomial_sum(monkeypatch):
    """64 monomials over a 2**6 coefficient: per-monomial work exceeds the
    ordered fold, so the call declines and the coefficient materializes."""
    rng = np.random.default_rng(9700)
    n = 2
    term = _factorized_term(n, "signed_sum")
    many = Sum(tuple(term.coupling_sum.terms) * 16)
    term = replace(term, coupling_sum=many)
    values = {"F": -1j * _signed(rng, (n, n, n)), "G": _signed(rng, (n, n, n))}
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
