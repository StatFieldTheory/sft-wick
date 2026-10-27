"""Opt-in dense contraction: admission rules and agreement with the ordered fold.

The cost rule declines these small fixtures, so ``admit_all`` bypasses it
wherever a test exercises the contraction itself.
"""

from dataclasses import replace
import pickle

import numpy as np
import pytest

import sft_wick._dense_contraction as dense
from sft_wick._dense_contraction import (
    plan_contraction,
    try_static_contraction,
    validate_contraction_options,
)
from sft_wick.expressions import Propagator, Rational, Symbol
from sft_wick.perturbation import DiagramTerm

from . import _contraction_exact as exact

LIMIT = 64 * 1024**2


@pytest.fixture
def admit_all(monkeypatch):
    """Run the backend on small fixtures, whatever the cost rule estimates."""
    monkeypatch.setattr(dense, "backend_is_cheaper", lambda backend, ordered: True)


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


def _check_bound(ig, cb, rp, jac, rb, *results):
    table = exact.dense_coefficients(ig.coupling_array, ig._e_psi)
    reference, scale = exact.exact_sum(
        table, ig.diagram_term.propagator_indices, ig.fixed_indices, cb, rp, jac, rb
    )
    factors = 1 + len(cb) + len(rb) + 2
    for values in results:
        exact.assert_within_bound(
            values, reference, scale, factors, ig.coupling_array.size
        )


@pytest.mark.parametrize("n", [2, 8])
@pytest.mark.parametrize("e_psi", [0, 1, 2, 3])
@pytest.mark.parametrize("matrix_r", [False, True])
def test_dense_real_action_matches_actual_diagram_term(admit_all, n, e_psi, matrix_r):
    ig, cb, rp, jac, rb = _case(n, e_psi=e_psi, matrix_r=matrix_r)
    all_arrays = [ig.coupling_array, rp, jac, cb[0][0]]
    all_arrays.extend(a for batch in rb for a in (batch[0], batch[3]))
    before = [(a.shape, a.dtype, a.tobytes()) for a in all_arrays]
    result = _candidate(ig, cb, rp, jac, rb)
    assert type(result) is np.ndarray and result.dtype == np.float64
    np.testing.assert_allclose(
        result, _ordered(ig, cb, rp, jac, rb), rtol=1e-12, atol=1e-14
    )
    _check_bound(ig, cb, rp, jac, rb, result)
    assert before == [(a.shape, a.dtype, a.tobytes()) for a in all_arrays]
    # The bounded structural plan cache holds shapes and labels only.
    assert plan_contraction.cache_info().maxsize == 4096


def test_pins_repeated_labels_and_propagator_assignment_wins(admit_all):
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


def test_native_readonly_strided_views_and_pickled_integrand(admit_all):
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
        np.array([[1 + 1e-6j, 2], [3, 4]]),
        np.array([[np.nan, 1], [2, 3]], dtype=np.float64),
        np.array([[np.inf, 1], [2, 3]], dtype=np.float64),
        np.array([[1, 1], [2, np.nan + 0j]]),
        np.ones((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=object),
    ],
)
def test_coefficient_phase_errors_and_nonfinite_values_decline(admit_all, coefficient):
    ig, cb, rp, jac, _ = _case()
    ig = replace(ig, coupling_array=coefficient)
    assert _candidate(ig, cb, rp, jac) is None


@pytest.mark.parametrize(
    "coefficient",
    [
        np.array([[0, 1], [2, 3]], dtype=np.float64),
        np.array([[0, 0], [0, 0]], dtype=np.complex128),
        np.array([[1 + 1e-15j, 2], [3, -4 - 1e-14j]]),
        np.array([[1e-300j, 2], [3, 4]]),
    ],
)
def test_zero_and_tolerated_imaginary_coefficients_follow_the_ordered_projector(
    admit_all, coefficient
):
    """Zero entries contribute exact zeros; imaginary residues the ordered
    projector accepts (below 1e-9 relative, or below the 1e-290 floor) are
    dropped exactly as it drops them."""
    ig, cb, rp, jac, _ = _case()
    ig = replace(ig, coupling_array=coefficient)
    result = _candidate(ig, cb, rp, jac)
    assert result is not None
    _check_bound(ig, cb, rp, jac, [], result, _ordered(ig, cb, rp, jac))


def test_odd_external_phase_of_a_real_coefficient_declines(admit_all):
    ig, cb, rp, jac, _ = _case(e_psi=1)
    real = ig.coupling_array.imag.copy()
    assert np.all(real != 0.0)
    ig = replace(ig, coupling_array=real)
    assert _candidate(ig, cb, rp, jac) is None
    with pytest.raises(ValueError, match="imaginary part"):
        _ordered(ig, cb, rp, jac)


def test_wrong_phase_error_and_sparse_unused_nan_are_preserved(admit_all):
    ig, cb, rp, jac, _ = _case()
    wrong = replace(ig, coupling_array=np.array([[1j, -1j], [2j, -2j]]))
    assert _candidate(wrong, cb, rp, jac) is None
    with pytest.raises(ValueError, match="imaginary part"):
        _ordered(wrong, cb, rp, jac)
    zero = replace(ig, coupling_array=np.zeros((2, 2), dtype=np.complex128))
    unused_nan = [(np.full((5, 2, 2), np.nan), "i", "j")]
    assert _candidate(zero, unused_nan, rp, jac) is None
    np.testing.assert_array_equal(_ordered(zero, unused_nan, rp, jac), np.zeros(5))


@pytest.mark.parametrize("kind", ["float32", "complex", "nonfinite", "subclass"])
def test_unsupported_kernel_falls_back(admit_all, kind):
    ig, cb, rp, jac, _ = _case()
    matrix = cb[0][0]
    if kind == "float32":
        matrix = matrix.astype(np.float32)
    elif kind == "complex":
        matrix = matrix.astype(np.complex128)
    elif kind == "nonfinite":
        matrix = matrix.copy()
        matrix[0, 0, 0] = np.inf
    else:

        class Array(np.ndarray):
            pass

        matrix = matrix.view(Array)
    assert _candidate(ig, [(matrix, "i", "j")], rp, jac) is None


@pytest.mark.parametrize("kind", ["diagonal", "isotropic"])
def test_diagonal_and_index_free_kernels_are_contracted(admit_all, kind):
    ig, cb, rp, jac, _ = _case(n=3)
    matrix = cb[0][0]
    if kind == "diagonal":
        cb = [(matrix[:, 0, :], "i", "j")]
    else:
        scale = matrix[:, 0, 0]
        cb = [(matrix, "i", "j"), (scale[:, None, None] * np.eye(3), None, None)]
    result = _candidate(ig, cb, rp, jac)
    assert result is not None
    _check_bound(ig, cb, rp, jac, [], result, _ordered(ig, cb, rp, jac))


def test_non_isotropic_index_free_c_declines_and_ordered_raises(admit_all):
    ig, cb, rp, jac, _ = _case()
    cb = [(cb[0][0], "i", "j"), (cb[0][0], None, None)]
    assert _candidate(ig, cb, rp, jac) is None
    with pytest.raises(ValueError, match="not a multiple of the"):
        _ordered(ig, cb, rp, jac)


@pytest.mark.parametrize(
    "labels, pins",
    [
        (("missing", "j"), {}),
        (("1", "j"), {"1": 0}),
        (("a", "j"), {"a": -1}),
        (("a", "j"), {"a": 2}),
    ],
)
def test_unresolved_literal_and_out_of_range_indices_fall_back(admit_all, labels, pins):
    ig, cb, rp, jac, _ = _case()
    ig = replace(ig, fixed_indices=pins)
    assert _candidate(ig, [(cb[0][0], *labels)], rp, jac) is None


def test_partial_response_support_is_contracted(admit_all):
    ig, cb, rp, jac, rb = _case(matrix_r=True)
    response, support = rb[0][0].copy(), rb[0][3].copy()
    support[0, 0] = False
    response[:, 0, 0] = 0.0
    rb = [(response, "j", "a", support)]
    result = _candidate(ig, cb, rp, jac, rb)
    assert result is not None
    _check_bound(ig, cb, rp, jac, rb, result, _ordered(ig, cb, rp, jac, rb))


def test_response_entry_outside_its_support_declines(admit_all):
    """The ordered fold skips an unsupported entry whatever it holds."""
    ig, cb, rp, jac, rb = _case(matrix_r=True)
    support = rb[0][3].copy()
    support[0, 0] = False
    assert _candidate(ig, cb, rp, jac, [(rb[0][0], "j", "a", support)]) is None


@pytest.mark.parametrize(
    "scales",
    [(1e308, 1e-308, 1.0, 1.0), (1e200, 1e200, 1e-200, 1e-200)],
)
def test_possible_overflow_declines(admit_all, scales):
    ig, cb, rp, jac, _ = _case()
    ig = replace(ig, coupling_array=np.full((2, 2), scales[0], dtype=np.float64))
    cb = [(np.full((5, 2, 2), scales[1], dtype=np.float64), "i", "j")]
    assert _candidate(ig, cb, np.full(5, scales[2]), np.full(5, scales[3])) is None


@pytest.mark.parametrize(
    "scales",
    [
        (1e-300, 1.0, 1.0, 1.0),
        (1e-100, 1e-100, 1e-100, 1e-100),
        (np.nextafter(0.0, 1.0), 1.0, 1.0, 1.0),
    ],
)
def test_underflow_ranges_are_admitted(admit_all, scales):
    ig, cb, rp, jac, _ = _case()
    coefficient = np.full((2, 2), scales[0], dtype=np.float64)
    coefficient[0, 1] *= -1
    ig = replace(ig, coupling_array=coefficient)
    cb = [(cb[0][0] * scales[1], "i", "j")]
    rp, jac = rp * scales[2], jac * scales[3]
    result = _candidate(ig, cb, rp, jac)
    assert result is not None
    _check_bound(ig, cb, rp, jac, [], result)


def test_signed_cancellation_is_bounded_not_bitwise(admit_all):
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
    # Cancellation leaves both evaluation orders far from the exact 0.42 in
    # relative terms; each stays inside the bound, which scales with S.
    _check_bound(ig, cb, rp, jac, [], result, _ordered(ig, cb, rp, jac))


def test_underflow_error_policy_controls_admission(admit_all):
    ig, cb, rp, jac, _ = _case()
    for policy in ({"all": "raise"}, {"under": "warn"}, {"under": "raise"}):
        with np.errstate(**policy):
            assert _candidate(ig, cb, rp, jac) is None
    # Finite inputs below the overflow bound cannot raise any other event.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        assert _candidate(ig, cb, rp, jac) is not None


def test_budget_declines_before_numeric_work(admit_all, monkeypatch):
    ig, cb, rp, jac, _ = _case(8)

    def forbidden(*args, **kwargs):
        raise AssertionError("numeric inspection reached under a rejected budget")

    for name in ("finite_maximum", "rotated_real_coefficient", "execute"):
        monkeypatch.setattr(dense, name, forbidden)
    assert _candidate(ig, cb, rp, jac, limit=1) is None


def test_plan_counts_are_exact_integers_and_cache_is_bounded():
    plan_contraction.cache_clear()
    for samples in range(1, 40):
        plan = plan_contraction(
            ((2, 2), (2, 2, samples), (samples,), (samples,)),
            ((1, 2), (1, 2, 0), (0,), (0,)),
        )
        assert plan is not None
        steps, flops, workspace = plan
        assert len(steps) == 3 and steps[-1][2] == (0,)
        assert type(flops) is int and type(workspace) is int and workspace >= samples
    assert plan_contraction.cache_info().currsize == 39
    assert plan_contraction(((2,), (3, 4)), ((1,), (1, 0))) is None


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
def test_extended_native_dtype_aliases_are_not_float64_backend(admit_all, field):
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


def test_cost_rule_declines_small_fixtures():
    """These fixtures are cheaper on the ordered fold than the backend's
    fixed overhead, so the default rule keeps the ordered fold."""
    ig, cb, rp, jac, rb = _case(matrix_r=True)
    assert _candidate(ig, cb, rp, jac, rb) is None
