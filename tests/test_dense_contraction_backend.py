"""Opt-in dense contraction admission and ordered-science compatibility."""

from dataclasses import replace
import pickle

import numpy as np
import pytest

from sft_wick._dense_contraction import (
    _plan,
    try_static_contraction,
    validate_contraction_options,
)
from sft_wick.expressions import Propagator, Rational, Symbol
from sft_wick.perturbation import DiagramTerm

LIMIT = 64 * 1024**2


def _case(n=2, samples=5, *, e_psi=0, matrix_r=False):
    rng = np.random.default_rng(720 + n)
    props = [Propagator("C", "i", "j", "x", "y")]
    if matrix_r:
        props.append(Propagator("R", "j", "a", "x", "z"))
    dt = DiagramTerm(
        tuple(props),
        Symbol("A", ("i", "j")),
        Rational(1),
        (),
        (("i", n), ("j", n)),
        int(matrix_r),
        n_external_response=e_psi,
    )
    values = rng.uniform(0.2, 1.2, (n, n))
    # build_integrand applies (-i)**n_response itself.
    cv = {"A": values * ((1j) ** (int(matrix_r) - e_psi))}
    integrand = dt.build_integrand(cv, fixed_indices={"a": n - 1})
    c = rng.uniform(0.1, 1.1, (samples, n, n))
    cb = [(c, "i", "j")]
    rb = []
    if matrix_r:
        r = rng.uniform(0.2, 0.9, (samples, n, n))
        rb = [(r, "j", "a", np.ones((n, n), dtype=bool))]
    rp = rng.uniform(0.1, 0.8, samples)
    jac = rng.uniform(0.2, 0.7, samples)
    return integrand, cb, rp, jac, rb


def _candidate(ig, cb, rp, jac, rb=(), limit=LIMIT):
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


def _ordered(ig, cb, rp, jac, rb=()):
    # Select the ordered backend explicitly, whatever the case was built with.
    ig = replace(ig, _contraction_backend="ordered")
    return ig._static_values(cb, rp, jac, r_batches=rb)


@pytest.mark.parametrize("n", [2, 8])
@pytest.mark.parametrize("e_psi", [0, 1, 2, 3])
@pytest.mark.parametrize("matrix_r", [False, True])
def test_dense_real_action_matches_actual_diagram_term(n, e_psi, matrix_r):
    ig, cb, rp, jac, rb = _case(n, e_psi=e_psi, matrix_r=matrix_r)
    all_arrays = [ig.coupling_array, rp, jac, cb[0][0]]
    all_arrays.extend(a for batch in rb for a in (batch[0], batch[3]))
    before = [(a.shape, a.dtype, a.tobytes()) for a in all_arrays]
    result = _candidate(ig, cb, rp, jac, rb)
    assert type(result) is np.ndarray and result.dtype == np.float64
    np.testing.assert_allclose(
        result, _ordered(ig, cb, rp, jac, rb), rtol=1e-12, atol=1e-14
    )
    assert before == [(a.shape, a.dtype, a.tobytes()) for a in all_arrays]
    # Structural cache contents do not retain a numeric input or integrand.
    assert _plan.cache_info().maxsize == 32


def test_pins_repeated_labels_and_propagator_assignment_wins():
    dt = DiagramTerm(
        (Propagator("C", "i", "i", "x", "y"), Propagator("C", "a", "i", "z", "y")),
        Symbol("A", ("i",)),
        Rational(1),
        (),
        (("i", 3),),
        0,
    )
    ig = dt.build_integrand({"A": np.array([0.4, 0.7, 1.1])}, fixed_indices={"a": 1})
    # This manual added pin collides with an existing summation name.  The
    # static ordered merge assigns i for each component after fixed_indices.
    ig = replace(ig, fixed_indices={"a": 1, "i": 99})
    matrix = (np.arange(18, dtype=np.float64).reshape(2, 3, 3) + 1) / 20
    cb = [(matrix, "i", "i"), (matrix.transpose(0, 2, 1), "a", "i")]
    rp = np.array([0.5, 0.6])
    jac = np.array([0.8, 0.9])
    result = _candidate(ig, cb, rp, jac)
    assert result is not None
    np.testing.assert_allclose(
        result, _ordered(ig, cb, rp, jac), rtol=1e-12, atol=1e-14
    )


def test_native_readonly_strided_views_and_pickled_integrand():
    ig, cb, rp, jac, rb = _case(matrix_r=True)
    underlying = np.empty((2, 4), dtype=np.complex128)
    underlying[:, ::2] = ig.coupling_array
    coefficient = underlying[:, ::2]
    coefficient.flags.writeable = False
    ig = replace(ig, coupling_array=coefficient)
    cb = [(cb[0][0].transpose(0, 2, 1), "i", "j")]
    cb[0][0].flags.writeable = False
    before = coefficient.tobytes(), cb[0][0].tobytes()
    candidate = _candidate(ig, cb, rp, jac, rb)
    assert candidate is not None and not np.shares_memory(candidate, coefficient)
    np.testing.assert_allclose(
        candidate, _ordered(ig, cb, rp, jac, rb), rtol=1e-12, atol=1e-14
    )
    clone = pickle.loads(pickle.dumps(ig))
    np.testing.assert_array_equal(candidate, _candidate(clone, cb, rp, jac, rb))
    assert before == (coefficient.tobytes(), cb[0][0].tobytes())


@pytest.mark.parametrize(
    "coefficient",
    [
        np.array([[1j, -1j], [2j, -2j]]),
        np.array([[1 + 1e-15j, 2], [3, 4]]),
        np.array([[0, 1], [2, 3]], dtype=np.float64),
        np.array([[np.nan, 1], [2, 3]], dtype=np.float64),
        np.array([[np.inf, 1], [2, 3]], dtype=np.float64),
        np.ones((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=object),
    ],
)
def test_coefficient_fallback_retains_legacy_phase_and_sparse_rules(coefficient):
    ig, cb, rp, jac, _ = _case()
    ig = replace(ig, coupling_array=coefficient)
    assert _candidate(ig, cb, rp, jac) is None


def test_wrong_phase_error_and_sparse_unused_nan_are_preserved():
    ig, cb, rp, jac, _ = _case()
    wrong = replace(ig, coupling_array=np.array([[1j, -1j], [2j, -2j]]))
    assert _candidate(wrong, cb, rp, jac) is None
    with pytest.raises(ValueError, match="imaginary part"):
        _ordered(wrong, cb, rp, jac)
    zero = replace(ig, coupling_array=np.zeros((2, 2), dtype=np.complex128))
    unused_nan = [(np.full((5, 2, 2), np.nan), "i", "j")]
    assert _candidate(zero, unused_nan, rp, jac) is None
    np.testing.assert_array_equal(_ordered(zero, unused_nan, rp, jac), np.zeros(5))


@pytest.mark.parametrize(
    "kind", ["float32", "complex", "nonfinite", "subclass", "diagonal", "trace"]
)
def test_unsupported_kernel_falls_back(kind):
    ig, cb, rp, jac, _ = _case()
    matrix = cb[0][0]
    if kind == "float32":
        matrix = matrix.astype(np.float32)
    elif kind == "complex":
        matrix = matrix.astype(np.complex128)
    elif kind == "nonfinite":
        matrix = matrix.copy()
        matrix[0, 0, 0] = np.inf
    elif kind == "subclass":

        class Array(np.ndarray):
            pass

        matrix = matrix.view(Array)
    elif kind == "diagonal":
        matrix = matrix[:, 0, :]
    else:
        cb = [(matrix, None, None)]
    if kind != "trace":
        cb = [(matrix, "i", "j")]
    assert _candidate(ig, cb, rp, jac) is None


@pytest.mark.parametrize(
    "labels, pins",
    [
        (("missing", "j"), {}),
        (("1", "j"), {"1": 0}),
        (("a", "j"), {"a": -1}),
        (("a", "j"), {"a": 2}),
    ],
)
def test_unresolved_literal_and_out_of_range_indices_fall_back(labels, pins):
    ig, cb, rp, jac, _ = _case()
    ig = replace(ig, fixed_indices=pins)
    assert _candidate(ig, [(cb[0][0], *labels)], rp, jac) is None


def test_matrix_response_support_must_be_full():
    ig, cb, rp, jac, rb = _case(matrix_r=True)
    support = rb[0][3].copy()
    support[0, 0] = False
    assert _candidate(ig, cb, rp, jac, [(rb[0][0], "j", "a", support)]) is None


@pytest.mark.parametrize(
    "scales",
    [
        (1e308, 1e-308, 1.0, 1.0),
        (1e200, 1e200, 1e-200, 1e-200),
        (1e-300, 1.0, 1.0, 1.0),
        (1e-100, 1e-100, 1e-100, 1e-100),
        (np.nextafter(0.0, 1.0), 1.0, 1.0, 1.0),
    ],
)
def test_extreme_subset_product_and_cancellation_lattice_ranges_decline(scales):
    ig, cb, rp, jac, _ = _case()
    ig = replace(ig, coupling_array=np.full((2, 2), scales[0], dtype=np.float64))
    cb = [(np.full((5, 2, 2), scales[1], dtype=np.float64), "i", "j")]
    assert _candidate(ig, cb, np.full(5, scales[2]), np.full(5, scales[3])) is None


def test_signed_cancellation_is_opt_in_rounding_not_bitwise_contract():
    dt = DiagramTerm(
        (Propagator("C", "i", "i", "x", "y"),),
        Symbol("A", ("i",)),
        Rational(1),
        (),
        (("i", 4),),
        0,
    )
    ig = dt.build_integrand({"A": np.array([1e16, 1.0, -1e16, 1.0])})
    cb = [(np.ones((2, 4, 4)), "i", "i")]
    rp, jac = np.full(2, 0.3), np.full(2, 0.7)
    result = _candidate(ig, cb, rp, jac)
    assert result is not None and np.all(np.isfinite(result))
    reference = _ordered(ig, cb, rp, jac)
    # A caller must choose tolerances based on the scientific query, including
    # cancellation.  This fixture rejects a claim of universal exact parity.
    assert not np.array_equal(reference, result)
    assert not np.allclose(reference, result, rtol=1e-12, atol=1e-14)


def test_nondefault_numpy_error_policy_declines_without_changed_warning():
    ig, cb, rp, jac, _ = _case()
    with np.errstate(all="raise"):
        assert _candidate(ig, cb, rp, jac) is None
    assert _candidate(ig, cb, rp, jac) is not None


def test_budget_declines_before_inspection_rotation_or_contraction(monkeypatch):
    import sft_wick._dense_contraction as dense

    ig, cb, rp, jac, _ = _case(8)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "numeric allocation/inspection reached under rejected budget"
        )

    monkeypatch.setattr(dense, "_scan", forbidden)
    monkeypatch.setattr(dense.np, "einsum", forbidden)
    assert _candidate(ig, cb, rp, jac, limit=1) is None


def test_path_intermediate_counts_are_exact_integers_and_cache_is_bounded():
    _plan.cache_clear()
    for samples in range(1, 40):
        plan = _plan(
            ((2, 2), (samples, 2, 2), (samples,), (samples,)),
            ((1, 2), (0, 1, 2), (0,), (0,)),
        )
        assert plan is not None
        path, total, maximum = plan
        assert path[0] == "einsum_path"
        assert type(total) is int and type(maximum) is int and maximum >= samples
    assert _plan.cache_info().currsize == 32


@pytest.mark.parametrize(
    "backend, limit",
    [
        ("fast", LIMIT),
        (None, LIMIT),
        ("ordered", 0),
        ("einsum", True),
        ("einsum", 1.5),
        ("einsum", np.int64(10)),
    ],
)
def test_options_validate_explicit_contract(backend, limit):
    with pytest.raises(ValueError):
        validate_contraction_options(backend, limit)


def test_default_and_opt_in_options_validate():
    validate_contraction_options("ordered", LIMIT)
    validate_contraction_options("einsum", LIMIT)


@pytest.mark.parametrize(
    "field",
    ["coefficient_real", "coefficient_complex", "C", "R", "scalar_R", "jacobian"],
)
def test_extended_native_dtype_aliases_are_not_float64_backend(field):
    ig, cb, rp, jac, rb = _case(matrix_r=True)
    if field.startswith("coefficient"):
        dtype = np.longdouble if field.endswith("real") else np.clongdouble
        ig = replace(ig, coupling_array=ig.coupling_array.real.astype(dtype))
    elif field == "C":
        cb = [(cb[0][0].astype(np.longdouble), *cb[0][1:])]
    elif field == "R":
        rb = [(rb[0][0].astype(np.longdouble), *rb[0][1:])]
    elif field == "scalar_R":
        rp = rp.astype(np.longdouble)
    else:
        jac = jac.astype(np.longdouble)
    assert _candidate(ig, cb, rp, jac, rb) is None


def test_custom_external_phase_property_keeps_ordered_reads(monkeypatch):
    from sft_wick.evaluate import DiagramIntegrand

    ig, cb, rp, jac, rb = _case()
    original = DiagramIntegrand.__dict__["_e_psi"]
    reads = []

    def phase(self):
        reads.append(self)
        return original.fget(self)

    monkeypatch.setattr(DiagramIntegrand, "_e_psi", property(phase))
    baseline = ig._static_values(cb, rp, jac, r_batches=rb)
    count = len(reads)
    reads.clear()
    candidate = replace(ig, _contraction_backend="einsum")._static_values(
        cb, rp, jac, r_batches=rb
    )
    np.testing.assert_array_equal(candidate, baseline)
    assert len(reads) == count
