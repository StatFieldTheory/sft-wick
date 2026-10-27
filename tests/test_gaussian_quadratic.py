"""Independent finite-Gaussian identities and adapter boundary checks."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import copy
from decimal import Decimal, localcontext
import gc
import itertools
import math
import pickle
import time
import weakref

import mpmath as mp
import numpy as np
import pytest

from sft_wick import Field, Product, Propagator, Rational, Sum, wick_contract
from sft_wick.quadratic import GaussianQuadratic, GaussianQuadraticFamily

C = np.array([[1.2, 0.25], [0.25, 0.7]])
A = np.array([[0.15, 0.08], [0.08, -0.12]])
ELL = np.array([0.4, -0.3])
MU = np.array([0.12, -0.17])
B = 0.2


def _gh(covariance=C, matrix=A, linear=ELL, mean=MU, constant=B):
    eigenvalues, rotation = np.linalg.eigh(covariance)
    factor = rotation @ np.diag(np.sqrt(eigenvalues))
    nodes, weights = np.polynomial.hermite.hermgauss(40)
    nodes, weights = nodes * np.sqrt(2), weights / np.sqrt(np.pi)
    normal = np.array(list(itertools.product(nodes, nodes)))
    probability = np.array([a * b for a, b in itertools.product(weights, weights)])
    z = normal @ factor.T + mean
    y = constant + z @ linear + np.einsum("ni,ij,nj->n", z, matrix, z)
    return probability, y


def _cumulants(raw):
    return np.array(
        [
            raw[1],
            raw[2] - raw[1] ** 2,
            raw[3] - 3 * raw[2] * raw[1] + 2 * raw[1] ** 3,
            raw[4]
            - 4 * raw[3] * raw[1]
            - 3 * raw[2] ** 2
            + 12 * raw[2] * raw[1] ** 2
            - 6 * raw[1] ** 4,
        ]
    )


def _partitions(items):
    if not items:
        yield []
        return
    first, *rest = items
    for partition in _partitions(rest):
        yield [(first,)] + partition
        for i in range(len(partition)):
            yield partition[:i] + [(first,) + partition[i]] + partition[i + 1 :]


def _connected(pairs, owners, count):
    edges = [set() for _ in range(count)]
    for left, right in pairs:
        a, b = owners[left], owners[right]
        edges[a].add(b)
        edges[b].add(a)
    seen, pending = {0}, [0]
    while pending:
        for other in edges[pending.pop()]:
            if other not in seen:
                seen.add(other)
                pending.append(other)
    return len(seen) == count


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_public_wick_connected_ring_chain_multiplicities(order):
    phi = Field("z", "physical")
    ring = chain = 0
    for degrees in itertools.product((1, 2), repeat=order):
        owners = [i for i, degree in enumerate(degrees) for _ in range(degree)]
        _, pairs = wick_contract([phi(str(i)) for i in range(len(owners))])
        connected = sum(_connected(pairing, owners, order) for pairing in pairs)
        if all(degree == 2 for degree in degrees):
            ring += connected
        elif degrees.count(1) == 2:
            chain += connected
    assert ring == 2 ** (order - 1) * math.factorial(order - 1)
    assert chain == (0 if order == 1 else math.factorial(order) * 2 ** (order - 3))
    law = GaussianQuadratic([[1.0]], [[0.3]], [0.7])
    expected = (
        0.3 if order == 1 else ring * 0.3**order + chain * 0.7**2 * 0.3 ** (order - 2)
    )
    assert law.cumulant(order) == pytest.approx(expected)


def test_public_matrix_wick_covariance_routing_matches_cumulants():
    phi = Field("z", "physical")
    cache = {(0, 0): 1.0}

    def evaluate(expression):
        if isinstance(expression, Rational):
            return expression.numerator / expression.denominator
        if isinstance(expression, Propagator):
            assert expression.kind == "C"
            return C[
                int(expression.spatial_left[1:]), int(expression.spatial_right[1:])
            ]
        if isinstance(expression, Product):
            return math.prod(evaluate(term) for term in expression.factors)
        if isinstance(expression, Sum):
            return sum(evaluate(term) for term in expression.terms)
        raise TypeError(type(expression))

    def monomial(alpha):
        if alpha not in cache:
            operators = [phi("z0") for _ in range(alpha[0])] + [
                phi("z1") for _ in range(alpha[1])
            ]
            expression, _ = wick_contract(operators)
            cache[alpha] = evaluate(expression)
        return cache[alpha]

    shifted, linear = B + ELL @ MU + MU @ A @ MU, ELL + 2 * A @ MU
    observable = {
        (0, 0): shifted,
        (1, 0): linear[0],
        (0, 1): linear[1],
        (2, 0): A[0, 0],
        (1, 1): 2 * A[0, 1],
        (0, 2): A[1, 1],
    }
    power, raw = {(0, 0): 1.0}, []
    for _ in range(5):
        raw.append(sum(value * monomial(alpha) for alpha, value in power.items()))
        following = {}
        for left, a in power.items():
            for right, b in observable.items():
                alpha = tuple(i + j for i, j in zip(left, right))
                following[alpha] = following.get(alpha, 0.0) + a * b
        power = following
    law = GaussianQuadratic(C, A, ELL, constant=B, mean=MU)
    np.testing.assert_allclose(
        [law.cumulant(n) for n in range(1, 5)], _cumulants(raw), rtol=3e-13, atol=1e-14
    )


@pytest.mark.parametrize("source", [-0.35, 0.25, 0.2 + 0.3j, 1j * 0.9])
def test_matrix_cgf_independent_gauss_hermite(source):
    probability, observation = _gh()
    law = GaussianQuadratic(C, A, ELL, constant=B, mean=MU)
    independent = np.log(probability @ np.exp(source * observation))
    assert law.cgf(source) == pytest.approx(independent, rel=2e-13, abs=1e-14)


def test_gauss_hermite_cumulants_and_taylor_coefficients():
    probability, observation = _gh()
    law = GaussianQuadratic(C, A, ELL, constant=B, mean=MU)
    independent = _cumulants([probability @ observation**n for n in range(5)])
    np.testing.assert_allclose(
        [law.cumulant(n) for n in range(1, 5)], independent, atol=2e-14
    )
    coefficients = law.taylor_coefficients(4)
    np.testing.assert_allclose(
        coefficients[1:], independent / [1, 2, 6, 24], atol=2e-14
    )
    assert coefficients[0] == 0
    assert law.cgf(0.001) == pytest.approx(
        np.polynomial.polynomial.polyval(0.001, coefficients), abs=1e-15
    )


def test_centering_tiny_sources_and_nonzero_mean():
    law = GaussianQuadratic(C, A, ELL, constant=B, mean=MU)
    assert law.cumulant(1, centered=True) == 0
    assert law.taylor_coefficients(4, centered=True)[1] == 0
    for source in (1e-8, -1e-8, 1e-8j):
        expected = law.variance * source**2 / 2
        assert law.cgf(source, centered=True) == pytest.approx(
            expected, rel=1e-7, abs=1e-32
        )
    assert law.cgf(0.2, centered=True) == pytest.approx(
        law.cgf(0.2) - 0.2 * law.mean, abs=1e-14
    )


def test_orthogonal_coordinate_and_latent_factor_invariance():
    angle = 0.731
    q = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    # The factor route is invariant without introducing strict-covariance roundoff asymmetry.
    factor = np.linalg.cholesky(C)
    original = GaussianQuadratic.from_factor(factor, A, ELL, constant=B, mean=MU)
    transformed = GaussianQuadratic.from_factor(
        q.T @ factor, q.T @ A @ q, q.T @ ELL, constant=B, mean=q.T @ MU
    )
    rotated_latent = GaussianQuadratic.from_factor(
        factor @ q, A, ELL, constant=B, mean=MU
    )
    sources = np.array([-0.3, 0.2, 0.3j])
    np.testing.assert_allclose(
        transformed.cgf(sources), original.cgf(sources), atol=3e-15
    )
    np.testing.assert_allclose(
        rotated_latent.cgf(sources), original.cgf(sources), atol=3e-15
    )


def test_psd_factor_zero_covariance_and_small_positive_modes():
    law = GaussianQuadratic.from_factor(
        [[1.0], [2.0]], np.eye(2), [1.0, -1.0], mean=[0.2, 0.3]
    )
    assert law.latent_dimension == 1
    assert law.variance > 0
    diagonal = GaussianQuadratic(np.diag([1.0, 0.0, 1e-30]), np.eye(3))
    assert diagonal.latent_dimension == 2
    assert 1e-30 in diagonal.eigenvalues
    zero = GaussianQuadratic(np.zeros((2, 2)), A, ELL, constant=B, mean=MU)
    assert zero.latent_dimension == 0
    assert zero.variance == 0
    assert zero.real_domain == (-math.inf, math.inf)
    assert zero.cgf(1e300, centered=True) == 0
    assert zero.cgf(0.5) == pytest.approx(0.5 * (B + ELL @ MU + MU @ A @ MU))
    explicit_zero = GaussianQuadratic.from_factor(np.empty((2, 0)), A, ELL)
    assert explicit_zero.cgf([0.0, 1.0, 1j]).shape == (3,)


def test_antisymmetric_quadratic_part_vanishes():
    law = GaussianQuadratic(C, A, ELL)
    altered = GaussianQuadratic(C, A + np.array([[0.0, 0.123], [-0.123, 0.0]]), ELL)
    np.testing.assert_allclose(
        law.cgf([-0.2, 0.3j]), altered.cgf([-0.2, 0.3j]), atol=1e-15
    )


@pytest.mark.parametrize(
    "covariance", [np.diag([1.0, -1e-30]), [[1.0, 0.2], [0.2000000000001, 1.0]]]
)
def test_strict_covariance_rejection_without_clipping(covariance):
    with pytest.raises(ValueError):
        GaussianQuadratic(covariance, np.eye(2))


@pytest.mark.parametrize("source", [-2.0, 1.0, -2.1, 1.1, -2.0 + 0.2j, 1.0 + 0.2j])
def test_real_poles_and_complex_strip_are_rejected(source):
    law = GaussianQuadratic.from_factor(np.eye(2), np.diag([-0.25, 0.5]))
    assert law.real_domain == (-2.0, 1.0)
    with pytest.raises(ValueError):
        law.cgf(source)


def test_indefinite_domain_near_poles_and_pure_linear_domain():
    law = GaussianQuadratic.from_factor(np.eye(2), np.diag([-0.25, 0.5]))
    for source in (np.nextafter(-2.0, 0.0), np.nextafter(1.0, 0.0)):
        assert np.isfinite(law.cgf(source))
    linear = GaussianQuadratic.from_factor(
        np.eye(2), np.zeros((2, 2)), [1.0, 2.0], constant=3.0
    )
    assert linear.real_domain == (-math.inf, math.inf)
    assert linear.cgf(0.7) == pytest.approx(3 * 0.7 + 2.5 * 0.7**2)


@pytest.mark.parametrize("dimension", [3, 8])
def test_characteristic_branch_uses_individual_logs(dimension):
    law = GaussianQuadratic.from_factor(np.eye(dimension), 0.5 * np.eye(dimension))
    omega = 10.0
    expected = -dimension / 2 * np.log(1 - 1j * omega)
    assert law.log_characteristic(omega) == pytest.approx(expected)
    assert law.characteristic(omega) == pytest.approx(np.exp(expected))
    assert (
        abs(
            law.log_characteristic(omega)
            + 0.5
            * np.log(np.linalg.det(np.eye(dimension) - 1j * omega * np.eye(dimension)))
        )
        > 1
    )


def test_characteristic_conjugacy_modulus_and_zero():
    law = GaussianQuadratic(C, A, ELL, constant=B, mean=MU)
    frequencies = np.linspace(0, 100, 101)
    np.testing.assert_allclose(
        law.characteristic(-frequencies),
        law.characteristic(frequencies).conj(),
        atol=1e-15,
    )
    assert law.characteristic(0.0) == 1
    assert np.all(np.abs(law.characteristic(frequencies)) <= 1 + 1e-15)
    assert np.all(law.log_characteristic(frequencies).real <= 1e-15)


def test_bounded_source_chunks_shapes_and_no_repeated_factorization(monkeypatch):
    calls = []
    original_eigh = np.linalg.eigh

    def eigh(matrix):
        calls.append(matrix.shape)
        return original_eigh(matrix)

    monkeypatch.setattr(np.linalg, "eigh", eigh)
    law = GaussianQuadratic.from_factor(
        np.eye(3), np.diag([-0.2, 0.0, 0.1]), [1.0, 0.0, -0.5]
    )
    assert len(calls) == 1
    rows = []
    original_evaluate = GaussianQuadratic._evaluate_chunk

    def evaluate(self, source, *, centered):
        rows.append(len(source))
        return original_evaluate(self, source, centered=centered)

    monkeypatch.setattr(GaussianQuadratic, "_evaluate_chunk", evaluate)
    sources = np.linspace(-0.5, 0.5, 35).reshape(5, 7)
    result = law.cgf(sources, chunk_size=4)
    assert result.shape == sources.shape
    assert max(rows) <= 4
    np.testing.assert_allclose(result, law.cgf(sources, chunk_size=100), atol=0.0)
    assert law.cgf(np.empty((0, 3))).shape == (0, 3)
    assert len(calls) == 1


def test_input_and_prepared_arrays_are_deep_snapshots():
    factor, quadratic, linear, mean = np.eye(2), A.copy(), ELL.copy(), MU.copy()
    law = GaussianQuadratic.from_factor(
        factor, quadratic, linear, constant=B, mean=mean
    )
    expected = law.cgf(0.2)
    factor[:] = quadratic[:] = linear[:] = mean[:] = 100
    assert law.cgf(0.2) == expected
    with pytest.raises(FrozenInstanceError):
        law._constant = 99
    with pytest.raises(ValueError):
        law.eigenvalues[0] = 99
    with pytest.raises(ValueError):
        law.eigenvalues.setflags(write=True)


@pytest.mark.parametrize("source", [0.7, -0.7, 0.7j, 0.3 + 0.4j])
def test_opt_in_gaussian_tail_preserves_mean_variance_and_bounds_cgf(source):
    law = GaussianQuadratic.from_factor(
        np.eye(4), np.diag([0.2, 0.03, -0.02, 0.0]), [0.5, 0.4, -0.2, 0.3], constant=0.1
    )
    approximate = law.compress(1)
    assert approximate.retained_dimension == 1
    assert approximate.spectral_tail.modes == 3
    assert approximate.mean == pytest.approx(law.mean, abs=1e-15)
    assert approximate.variance == pytest.approx(law.variance, abs=1e-15)
    assert (
        abs(law.cgf(source) - approximate.cgf(source))
        <= approximate.cgf_error_bound(source) + 2e-15
    )
    assert approximate.reference_real_domain == law.real_domain
    assert abs(approximate.characteristic(0.7)) <= 1


def test_compressed_object_does_not_retain_full_modes_or_parent():
    law = GaussianQuadratic.from_factor(
        np.eye(20), np.diag(np.linspace(-0.2, 0.2, 20)), np.ones(20)
    )
    parent_eigenvalues = weakref.ref(law._eigenvalues)
    parent_linear = weakref.ref(law._linear_squares)
    approximate = law.compress(2)
    expected = approximate.cgf(0.2)
    del law
    gc.collect()
    assert parent_eigenvalues() is None and parent_linear() is None
    assert approximate.cgf(0.2) == expected
    assert approximate._eigenvalues.nbytes + approximate._linear_squares.nbytes == 32


def test_compression_zero_modes_bound_domains_and_repeated_compression():
    law = GaussianQuadratic.from_factor(np.eye(2), np.zeros((2, 2)), [1.0, 2.0])
    approximate = law.compress(0)
    assert approximate.cgf_error_bound(1e300j) == 0
    assert approximate.cgf(0.2) == pytest.approx(law.cgf(0.2))
    assert law.compress(2) is law
    with pytest.raises(ValueError):
        law.cgf_error_bound(0.2)
    with pytest.raises(ValueError):
        approximate.compress(0)
    positive = GaussianQuadratic.from_factor(np.eye(2), np.diag([0.4, 0.1])).compress(0)
    assert positive.real_domain == (-math.inf, math.inf)
    with pytest.raises(ValueError):
        positive.cgf_error_bound(1.25)
    with pytest.raises(ValueError):
        positive.cgf_error_bound(1.25j)


def test_weighted_colored_and_white_conventions_are_distinct_and_exact():
    weights, intensity = np.array([0.2, 0.7]), np.array([1.2, 0.4])
    direct = GaussianQuadratic(
        C, weights[:, None] * A * weights[None, :], weights * ELL, constant=B, mean=MU
    )
    weighted = GaussianQuadratic.from_weighted_grid(
        C, A, weights, ELL, constant=B, mean=MU
    )
    np.testing.assert_allclose(
        direct.cgf([-0.3, 0.4j]), weighted.cgf([-0.3, 0.4j]), atol=1e-15
    )
    white = GaussianQuadratic.from_white_noise(
        intensity, A, weights, ELL, constant=B, mean=MU
    )
    nodal = GaussianQuadratic(
        np.diag(intensity / weights),
        weights[:, None] * A * weights[None, :],
        weights * ELL,
        constant=B,
        mean=MU,
    )
    np.testing.assert_allclose(
        white.cgf([-0.3, 0.4j]), nodal.cgf([-0.3, 0.4j]), atol=1e-15
    )
    whitened = GaussianQuadratic.from_factor(
        np.eye(2),
        np.sqrt(weights * intensity)[:, None]
        * A
        * np.sqrt(weights * intensity)[None, :],
        np.sqrt(weights * intensity) * ELL,
    )
    np.testing.assert_allclose(
        whitened.cgf([-0.3, 0.4j]),
        GaussianQuadratic.from_white_noise(intensity, A, weights, ELL).cgf(
            [-0.3, 0.4j]
        ),
        atol=1e-15,
    )
    assert GaussianQuadratic.from_white_noise([0.0, 0.0], A, weights).variance == 0


def test_joint_noncommuting_gauss_hermite_and_repeated_slot_coefficients():
    second = np.array([[0.05, -0.11], [-0.11, 0.23]])
    linear = np.array([ELL, [-0.15, 0.21]])
    family = GaussianQuadraticFamily(
        C, [A, second], linear, constant=[B, -0.07], mean=MU
    )
    probability, first_y = _gh()
    _, second_y = _gh(matrix=second, linear=linear[1], constant=-0.07)
    y = np.array([first_y, second_y])
    assert (
        np.linalg.norm(
            family._matrices[0] @ family._matrices[1]
            - family._matrices[1] @ family._matrices[0]
        )
        > 0.001
    )
    for slots in ((0, 1), (0, 0, 1), (0, 1, 1), (0, 0, 1, 1)):
        independent = 0.0
        for partition in _partitions(list(range(len(slots)))):
            value = math.prod(
                probability @ np.prod(y[[slots[i] for i in block]], axis=0)
                for block in partition
            )
            independent += (
                (-1) ** (len(partition) - 1)
                * math.factorial(len(partition) - 1)
                * value
            )
        assert family.mixed_cumulant(slots) == pytest.approx(independent, abs=2e-14)
    assert family.joint_coefficient([2, 2]) == pytest.approx(
        family.mixed_cumulant([0, 0, 1, 1]) / 4
    )
    assert family.joint_coefficient([0, 0]) == 0
    direction = np.array([0.2, -0.3])
    projected = family.project(direction)
    assert family.cgf(direction) == pytest.approx(
        np.log(probability @ np.exp(direction @ y)), abs=1e-14
    )
    assert family.log_characteristic(direction) == pytest.approx(projected.cgf(1j))
    assert family.mixed_cumulant([0] * 4) == pytest.approx(
        family.project([1.0, 0.0]).cumulant(4)
    )


def test_joint_projection_snapshot_zero_direction_and_order_gate():
    factor, matrices = np.eye(2), np.array([A, np.eye(2)])
    family = GaussianQuadraticFamily.from_factor(factor, matrices)
    projected = family.project([1.0, -0.2])
    expected = projected.cgf(0.2)
    factor[:] = matrices[:] = 100
    assert projected.cgf(0.2) == expected
    assert family.project([0.0, 0.0]).cgf([0.2, 1j]).tolist() == [0j, 0j]
    assert family.mixed_cumulant([0]) == pytest.approx(np.trace(A))
    with pytest.raises(ValueError):
        family._matrices.setflags(write=True)
    for indices in ([], [0] * 5, [2], [True]):
        with pytest.raises(ValueError):
            family.mixed_cumulant(indices)
    with pytest.raises(ValueError):
        family.joint_coefficient([3, 2])
    with pytest.raises(ValueError):
        family.cgf([1j, 0.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"covariance": [[1j]], "quadratic": [[1.0]]},
        {"covariance": [[np.nan]], "quadratic": [[1.0]]},
        {"covariance": [[1.0]], "quadratic": [[np.inf]]},
        {"covariance": [[1.0]], "quadratic": [[1.0]], "linear": [1j]},
        {"covariance": [[1.0]], "quadratic": [[1.0]], "mean": [np.nan]},
        {"covariance": [[1.0]], "quadratic": [[1.0]], "constant": True},
        {"covariance": [[1.0, 0.0]], "quadratic": [[1.0]]},
        {"covariance": [[1.0]], "quadratic": [[1.0, 0.0], [0.0, 1.0]]},
    ],
)
def test_invalid_observable_inputs(kwargs):
    with pytest.raises(ValueError):
        GaussianQuadratic(**kwargs)


@pytest.mark.parametrize(
    "operation",
    [
        lambda law: law.cgf(np.nan),
        lambda law: law.cgf([1.0, np.inf]),
        lambda law: law.cgf(0.1, chunk_size=0),
        lambda law: law.cgf(0.1, chunk_size=True),
        lambda law: law.cgf(0.1, centered=1),
        lambda law: law.log_characteristic(1j),
        lambda law: law.cumulant(0),
        lambda law: law.cumulant(True),
        lambda law: law.taylor_coefficients(-1),
        lambda law: law.compress(3),
        lambda law: law.compress(-1),
        lambda law: law.compress(True),
    ],
)
def test_invalid_query_inputs(operation):
    law = GaussianQuadratic.from_factor(np.eye(2), A)
    with pytest.raises(ValueError):
        operation(law)


@pytest.mark.parametrize(
    "weights,intensity",
    [([0.0, 1.0], [1.0, 1.0]), ([-1.0, 1.0], [1.0, 1.0]), ([1.0, 1.0], [-1.0, 1.0])],
)
def test_invalid_weighted_inputs(weights, intensity):
    with pytest.raises(ValueError):
        GaussianQuadratic.from_white_noise(intensity, A, weights)


def test_unrepresentable_queries_fail_explicitly():
    linear = GaussianQuadratic.from_factor([[1.0]], [[0.0]], [1.0])
    with pytest.raises(ValueError):
        linear.cgf(1e300)
    quadratic = GaussianQuadratic.from_factor([[1.0]], [[0.3]])
    with pytest.raises(OverflowError):
        quadratic.cumulant(200)


@pytest.mark.parametrize(
    "eigenvalue,endpoint,order",
    [(0.001, 0.7, 500), (-0.001, 0.7, 501), (0.01, 0.0, 200)],
)
def test_high_order_cumulants_combine_factors_before_float_conversion(
    eigenvalue, endpoint, order
):
    law = GaussianQuadratic.from_factor([[1.0]], [[eigenvalue]], [endpoint])
    with mp.workdps(100):
        lam, h = mp.mpf(eigenvalue), mp.mpf(endpoint)
        independent = mp.power(2, order - 1) * mp.factorial(
            order - 1
        ) * lam**order + mp.factorial(order) * mp.power(
            2, order - 3
        ) * h**2 * lam ** (
            order - 2
        )
    expected = float(independent)
    assert expected != 0 and np.isfinite(expected)
    assert law.cumulant(order) == pytest.approx(expected, rel=3e-12, abs=0.0)


def test_high_order_taylor_uses_scaled_power_identity():
    law = GaussianQuadratic.from_factor([[1.0]], [[0.5]], [0.7])
    coefficients = law.taylor_coefficients(1100)
    # lambda=1/2 makes the quadratic coefficient exactly 1/(2*n);
    # the chain coefficient is h**2/2 independently of n >= 2.
    assert coefficients[1100] == pytest.approx(
        1 / 2200 + 0.7**2 / 2, rel=2e-14, abs=0.0
    )
    pure = GaussianQuadratic.from_factor([[1.0]], [[0.5]])
    assert pure.taylor_coefficients(1100)[1100] == pytest.approx(
        1 / 2200, rel=2e-14, abs=0.0
    )


def test_high_order_zero_cancellation_and_true_final_underflow():
    symmetric = GaussianQuadratic.from_factor(
        np.eye(2), np.diag([-0.5, 0.5]), [0.7, 0.7]
    )
    assert symmetric.cumulant(1101) == 0
    assert symmetric.taylor_coefficients(1101)[1101] == 0
    zero = GaussianQuadratic.from_factor([[1.0]], [[0.0]], [0.7])
    assert zero.cumulant(10000) == 0
    tiny = GaussianQuadratic.from_factor([[1.0]], [[1e-5]])
    with mp.workdps(100):
        independent = mp.power(2, 499) * mp.factorial(499) * mp.mpf(1e-5) ** 500
    assert float(independent) == 0
    assert tiny.cumulant(500) == 0


def test_exact_dominant_opposite_groups_preserve_a_small_finite_remainder():
    law = GaussianQuadratic.from_factor(np.eye(3), np.diag([-100.0, 100.0, 0.001]))
    with mp.workdps(100):
        independent = mp.power(2, 100) * mp.factorial(100) * mp.mpf(0.001) ** 101
    assert float(independent) != 0
    assert law.cumulant(101) == pytest.approx(float(independent), rel=3e-12, abs=0.0)
    taylor = GaussianQuadratic.from_factor(np.eye(3), np.diag([-100.0, 100.0, 0.5]))
    assert taylor.taylor_coefficient(1101) == pytest.approx(
        1 / 2202, rel=3e-13, abs=0.0
    )
    with pytest.raises(OverflowError):
        taylor.taylor_coefficient(1100)


def test_single_taylor_coefficient_handles_zero_centering_and_variance_scaling():
    law = GaussianQuadratic.from_factor([[1.0]], [[1e154]])
    assert law.taylor_coefficient(0) == 0
    assert law.taylor_coefficient(1, centered=True) == 0
    assert law.taylor_coefficient(2) == pytest.approx(1e308, rel=3e-13, abs=0.0)
    with pytest.raises(OverflowError):
        law.cumulant(2)


@pytest.mark.parametrize("larger,order", [(1e100, 3), (0.5, 501)])
def test_distinct_adjacent_mirrors_keep_finite_decimal_residual(larger, order):
    smaller = float(np.nextafter(larger, 0.0))
    law = GaussianQuadratic.from_factor(np.eye(2), np.diag([larger, -smaller]))
    with localcontext() as context:
        context.prec = 180
        difference = Decimal(larger) ** order - Decimal(smaller) ** order
        coefficient = Decimal(2) ** (order - 1) * difference / Decimal(order)
        cumulant = (
            Decimal(2) ** (order - 1) * Decimal(math.factorial(order - 1)) * difference
        )
    assert float(coefficient) != 0 and np.isfinite(float(coefficient))
    assert law.taylor_coefficient(order) == pytest.approx(
        float(coefficient), rel=3e-13, abs=0.0
    )
    if abs(cumulant) <= Decimal(float(np.finfo(np.float64).max)):
        assert law.cumulant(order) == pytest.approx(float(cumulant), rel=3e-13, abs=0.0)
    else:
        with pytest.raises(OverflowError):
            law.cumulant(order)


@pytest.mark.parametrize(
    "transformation",
    [copy.copy, copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))],
)
def test_copy_deepcopy_pickle_keep_original_compressed_and_family_immutable(
    transformation,
):
    law = GaussianQuadratic.from_factor(np.eye(2), np.diag([0.25, -0.03]), [0.1, 0.2])
    for original in (law, law.compress(1)):
        transformed = transformation(original)
        assert transformed.cgf(0.2) == original.cgf(0.2)
        assert transformed.real_domain == original.real_domain
        assert transformed.reference_real_domain == original.reference_real_domain
        with pytest.raises(ValueError):
            transformed.eigenvalues[0] = 0.5
        with pytest.raises(ValueError):
            transformed.eigenvalues.setflags(write=True)
    family = GaussianQuadraticFamily.from_factor(np.eye(2), [A, np.eye(2)])
    transformed = transformation(family)
    assert transformed.cgf([0.1, -0.1]) == family.cgf([0.1, -0.1])
    with pytest.raises(ValueError):
        transformed._matrices.setflags(write=True)
    with pytest.raises(ValueError):
        transformed._endpoints.setflags(write=True)


@pytest.mark.parametrize("eigenvalue", [1e200, 1e-200])
def test_unrepresentable_tail_summary_refuses_compression_cleanly(eigenvalue):
    law = GaussianQuadratic.from_factor([[1.0]], [[eigenvalue]])
    assert law.cgf(0.0) == 0
    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="tail summaries"):
            law.compress(0)
    assert law.cgf(0.0) == 0


def test_large_source_small_spectral_scale_avoids_intermediate_powers():
    pure_linear = GaussianQuadratic.from_factor([[1.0]], [[0.0]], [1e-150])
    assert pure_linear.cgf(1e200) == pytest.approx(5e99, rel=3e-15, abs=0.0)
    law = GaussianQuadratic.from_factor([[1.0]], [[1e-150]])
    approximate = law.compress(0)
    source = 1e149
    bound = approximate.cgf_error_bound(source)
    assert np.isfinite(bound) and bound > 0
    assert abs(law.cgf(source) - approximate.cgf(source)) <= bound


def test_first_moments_sum_exact_mirrors_and_tiny_remainders_without_overflow():
    spectrum = np.array([8e307] * 3 + [-8e307] * 3 + [1.0])
    law = GaussianQuadratic.from_factor(np.eye(7), np.diag(spectrum), constant=0.25)
    family = GaussianQuadraticFamily.from_factor(
        np.eye(7), [np.diag(spectrum)], constant=[0.25]
    )
    with np.errstate(all="raise"):
        assert law.mean == 1.25
        assert law.cumulant(1) == 1.25
        assert law.taylor_coefficient(1) == 1.25
        assert family.mixed_cumulant([0]) == 1.25
    assert law.cgf(0.0) == 0
    assert np.isfinite(law.cgf(1e-309))
    mirrors = GaussianQuadratic.from_factor(np.eye(6), np.diag(spectrum[:6]))
    assert mirrors.mean == 0


def test_true_first_moment_overflow_raises_but_original_zero_source_is_valid():
    law = GaussianQuadratic.from_factor(np.eye(2), np.diag([1e308, 1e308]))
    family = GaussianQuadraticFamily.from_factor(np.eye(2), [np.diag([1e308, 1e308])])
    assert law.cgf(0.0) == 0
    with np.errstate(all="raise"):
        for operation in (
            lambda: law.mean,
            lambda: law.cumulant(1),
            lambda: law.taylor_coefficient(1),
            lambda: family.mixed_cumulant([0]),
        ):
            with pytest.raises(OverflowError):
                operation()


@pytest.mark.parametrize("endpoint", [1e-200, 2e-162, 1e-161])
def test_unrepresentable_squared_endpoint_rejects_and_rescaling_matches_oracle(
    endpoint,
):
    with pytest.raises(ValueError, match="underflows float64 storage"):
        GaussianQuadratic.from_factor([[1.0]], [[0.0]], [endpoint])
    # Rescale Y by 1e200 and scale its source by the reciprocal. This declared
    # law has representable spectral weights and the same exponential probe.
    rescaled = GaussianQuadratic.from_factor([[1.0]], [[0.0]], [1.0])
    with mp.workdps(100):
        cgf = mp.mpf("1e200") ** 2 * mp.mpf("1e-200") ** 2 / 2
        characteristic = mp.exp(-cgf)
    assert rescaled.cgf(1.0) == pytest.approx(float(cgf))
    assert rescaled.characteristic(1.0) == pytest.approx(float(characteristic))


def test_uncentered_tiny_source_preserves_centered_ring_and_signed_mean():
    spectrum = [-1e100, 1.0, 1e100]
    law = GaussianQuadratic.from_factor(np.eye(3), np.diag(spectrum))
    source = 1e-200
    with mp.workdps(450):
        t = mp.mpf(source)
        independent = -sum(mp.log1p(-2 * t * mp.mpf(lam)) for lam in spectrum) / 2
    assert float(independent) != 0
    assert law.mean == 1.0
    assert law.cgf(source) == pytest.approx(float(independent), rel=3e-14, abs=0.0)
    assert law.cgf(source, centered=True) == pytest.approx(
        float(independent - t), rel=3e-14, abs=0.0
    )
    sources = np.array([source, 0.0, -source])
    assert law.cgf(sources, chunk_size=1)[0] == law.cgf(source)
    assert law.cgf(sources, chunk_size=3)[0] == law.cgf(source)


def test_compression_keeps_exact_total_mean_after_tail_rounding_and_pickling():
    law = GaussianQuadratic.from_factor(np.eye(3), np.diag([-1e100, 1.0, 1e100]))
    approximate = law.compress(1)
    assert approximate.spectral_tail.mean == 1e100  # rounded diagnostic
    assert approximate.mean == law.mean == 1.0
    assert approximate.cumulant(1) == 1.0
    assert approximate.taylor_coefficient(1) == 1.0
    source = 1e-200
    with mp.workdps(450):
        t = mp.mpf(source)
        lam = mp.mpf(float(approximate.eigenvalues[0]))
        omitted_mean = sum(mp.mpf(value) for value in law.eigenvalues[1:])
        independent = (
            -mp.log1p(-2 * t * lam) / 2
            + omitted_mean * t
            + mp.mpf(approximate.spectral_tail.variance) * t**2 / 2
        )
    assert approximate.cgf(source) == pytest.approx(
        float(independent), rel=3e-14, abs=0.0
    )
    restored = pickle.loads(pickle.dumps(approximate))
    assert restored.mean == 1.0
    assert restored.cgf(source) == approximate.cgf(source)


def test_finite_source_product_of_unrepresentable_first_moment():
    law = GaussianQuadratic.from_factor(np.eye(2), np.diag([1e308, 1e308]))
    source = 1e-312
    with mp.workdps(100):
        independent = -mp.log1p(-2 * mp.mpf(source) * mp.mpf(1e308))
    with pytest.raises(OverflowError):
        _ = law.mean
    assert law.cgf(source) == pytest.approx(float(independent), rel=3e-14, abs=0.0)
    assert law.cgf(0.0) == 0.0
    frequency = source
    with mp.workdps(100):
        log_phi = -mp.log1p(-2j * mp.mpf(frequency) * mp.mpf(1e308))
    assert law.log_characteristic(frequency) == pytest.approx(
        complex(log_phi), rel=3e-14, abs=0.0
    )


def test_white_factor_avoids_ratio_underflow_and_preserves_nonzero_mean():
    # In the equivalent x=Wz representation, Var(x)=w*c=1; forming c/w
    # first would underflow to zero and incorrectly erase this random law.
    law = GaussianQuadratic.from_white_noise([1e-200], [[0.0]], [1e200], [1.0])
    with mp.workdps(100):
        variance = mp.mpf(1e200) * mp.mpf(1e-200)
        independent = variance / 2
    assert law.variance == pytest.approx(float(variance), rel=3e-14, abs=0.0)
    assert law.cgf(1.0) == pytest.approx(float(independent), rel=3e-14, abs=0.0)
    assert law.characteristic(1.0) == pytest.approx(
        complex(mp.exp(-independent)), rel=3e-14, abs=0.0
    )
    shifted = GaussianQuadratic.from_white_noise(
        [1e-200], [[0.25]], [1e200], [1.0], mean=[1e-200]
    )
    with mp.workdps(100):
        mu_x = mp.mpf(1e200) * mp.mpf(1e-200)
        expected_mean = mu_x + (mu_x**2 + variance) / 4
        expected_variance = (1 + mu_x / 2) ** 2 * variance + variance**2 / 8
    assert shifted.mean == pytest.approx(float(expected_mean), rel=3e-14, abs=0.0)
    assert shifted.variance == pytest.approx(
        float(expected_variance), rel=3e-14, abs=0.0
    )


def test_mixed_cumulants_recover_finite_remainders_after_dense_product_overflow():
    a = np.diag([1e150, -1e150, 1.0])
    b = np.diag([2e150, -2e150, 3.0])
    b += np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    assert not np.array_equal(a @ b, b @ a)
    family = GaussianQuadraticFamily.from_factor(np.eye(3), [a, b])
    with np.errstate(all="raise"):
        assert family.mixed_cumulant([0, 0, 0]) == 8.0
        assert family.joint_coefficient([3, 0]) == pytest.approx(4 / 3)
        # tr(A*A*B)=sum_i A_ii**2 B_ii=3. Off-diagonal B does not
        # contribute, independently of the matrices' noncommutation.
        assert family.mixed_cumulant([0, 0, 1]) == 24.0
        assert family.joint_coefficient([2, 1]) == 12.0


def test_mixed_adjacent_mirrors_use_decimal_oracle_before_final_conversion():
    larger = 1e100
    smaller = float(np.nextafter(larger, 0.0))
    a = np.diag([larger, -smaller, 1.0])
    b = np.diag([2 * larger, -2 * smaller, 3.0])
    family = GaussianQuadraticFamily.from_factor(np.eye(3), [a, b])
    with localcontext() as context:
        context.prec = 180
        pure = Decimal(8) * (Decimal(larger) ** 3 - Decimal(smaller) ** 3 + 1)
        mixed = Decimal(8) * (
            Decimal(2) * (Decimal(larger) ** 3 - Decimal(smaller) ** 3) + 3
        )
    assert family.mixed_cumulant([0, 0, 0]) == pytest.approx(
        float(pure), rel=3e-14, abs=0.0
    )
    assert family.joint_coefficient([3, 0]) == pytest.approx(
        float(pure / 6), rel=3e-14, abs=0.0
    )
    assert family.mixed_cumulant([0, 0, 1]) == pytest.approx(
        float(mixed), rel=3e-14, abs=0.0
    )
    assert family.joint_coefficient([2, 1]) == pytest.approx(
        float(mixed / 2), rel=3e-14, abs=0.0
    )


def test_joint_coefficient_can_be_finite_when_unscaled_cumulant_overflows():
    family = GaussianQuadraticFamily.from_factor([[1.0]], [[[1e154]]])
    with pytest.raises(OverflowError):
        family.mixed_cumulant([0, 0])
    assert family.joint_coefficient([2]) == pytest.approx(1e308, rel=3e-14, abs=0.0)


def test_mixed_exact_fallback_returns_padded_overflow_remainder():
    # Up to 0.7.0 the padded copy of the 3x3 case above was refused by a
    # 1e6-operation limit on a rational fallback. tr(A*A*B) = 3 exactly,
    # because 2 * fl(1e150) is fl(2e150); every one of the six words gives 3.
    dimension = 64
    a = np.zeros((dimension, dimension))
    b = a.copy()
    a[:3, :3] = np.diag([1e150, -1e150, 1.0])
    b[:3, :3] = np.diag([2e150, -2e150, 3.0])
    assert 2 * 1e150 == 2e150
    family = GaussianQuadraticFamily.from_factor(np.eye(dimension), [a, b])
    with np.errstate(all="raise"):
        assert family.mixed_cumulant([0, 0, 1]) == 24.0
        assert family.joint_coefficient([2, 1]) == 12.0
        assert family.mixed_cumulant([0, 0, 0]) == 8.0


def test_mixed_exact_fallback_has_explicit_work_bound():
    # Entries spread over 2000 binades need about 98 digit slices each at
    # N = 1536; the third-order exact contraction would take about 3e13
    # multiply-adds and 7 GiB, and is refused before any product. The
    # magnitudes also withhold the rounding bound, so no float64 value is kept.
    random = np.random.default_rng(889)
    dimension = 1536
    raw = random.normal(size=(dimension, dimension))
    raw *= np.exp2(random.integers(-1000, 1000, size=raw.shape).astype(np.float64))
    wide = raw + raw.T
    family = GaussianQuadraticFamily.from_factor(
        np.eye(dimension), np.array([wide, np.eye(dimension)])
    )
    started = time.perf_counter()
    with pytest.raises(ArithmeticError, match="exact contraction work bound"):
        family.mixed_cumulant([0, 0, 1])
    assert time.perf_counter() - started < 30.0


def test_family_projection_constants_preserve_a_small_signed_remainder():
    family = GaussianQuadraticFamily.from_factor(
        [[1.0]], np.zeros((3, 1, 1)), constant=[1e100, 1.0, -1e100]
    )
    projected = family.project([1.0, 1.0, 1.0])
    assert projected.mean == 1.0
    assert family.cgf([1.0, 1.0, 1.0]) == 1.0
    assert family.log_characteristic([1.0, 1.0, 1.0]) == 1j
    assert family.characteristic([1.0, 1.0, 1.0]) == pytest.approx(np.exp(1j))
    huge = GaussianQuadraticFamily.from_factor(
        [[1.0]], np.zeros((3, 1, 1)), constant=[1e308, 1.0, -1e308]
    )
    assert huge.project([1e100, 1.0, 1e100]).mean == 1.0
