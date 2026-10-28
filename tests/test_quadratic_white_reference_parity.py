"""Parity with a frozen reference implementation, and independent bounded-prefix witnesses."""

from __future__ import annotations

import ast
from fractions import Fraction
import hashlib
import inspect
import itertools
import math
from pathlib import Path
import struct
import sys
import tarfile
import types
import warnings

import numpy as np
import pytest

from sft_wick import _mixed_contraction as certificates
from sft_wick import quadratic
from sft_wick.white_cumulant import OUWhiteCumulant

from .mixed_reference import (
    assert_certified,
    float_statistic,
    fraction_statistic,
    integer_statistic,
    make_family,
)


@pytest.fixture(scope="module")
def frozen():
    archive = (
        Path(__file__).resolve().parents[1]
        / "tests/fixtures/quadratic_white_reference_source.tar.gz"
    )
    pins = {
        "quadratic": "eee500133e818e60cdda1c0ea268a8cf6f61a69d2ea4b0b6e7423187eaf7ed89",
        "white_cumulant": "83849f2081fad5b3e316699d8d48afd69f0f528176a7aae2e430703608088a66",
    }
    modules = {}
    with tarfile.open(archive) as stream:
        for short, expected in pins.items():
            member = stream.extractfile(f"src/sft_wick/{short}.py")
            assert member is not None
            source = member.read()
            assert hashlib.sha256(source).hexdigest() == expected
            name = f"frozen_reference_{short}"
            module = types.ModuleType(name)
            sys.modules[name] = module
            exec(compile(source, f"<{name}>", "exec"), module.__dict__)
            modules[short] = module
    return modules


def _outcome(call):
    try:
        result = call()
    except (ArithmeticError, ValueError, MemoryError) as error:
        return type(error), error.args
    if np.isscalar(result):
        return "float", struct.pack("!d", result)
    result = np.asarray(result)
    return result.shape, result.dtype.str, result.tobytes()


@pytest.mark.parametrize("order", [2, 3, 5])
@pytest.mark.parametrize("components", [1, 2])
@pytest.mark.parametrize("complex_amplitude", [False, True])
@pytest.mark.parametrize("samples", [0, 1, 127, 129, 257])
def test_ou_private_scratch_keeps_frozen_bytes_and_input_state(
    frozen, order, components, complex_amplitude, samples
):
    rng = np.random.default_rng(719 + order + samples)
    amplitude = rng.normal(size=(components,) * order)
    if complex_amplitude:
        amplitude = amplitude + 1j * rng.normal(size=amplitude.shape)
    rates = np.linspace(0.0, 1.7, components)
    times = rng.uniform(-0.3, 2.0, size=(order, samples))
    before = times.tobytes(), amplitude.tobytes(), rates.tobytes()
    arguments = dict(t_min=0.2, chunk_size=128)
    reference = frozen["white_cumulant"].OUWhiteCumulant(amplitude, rates, **arguments)
    candidate = OUWhiteCumulant(amplitude, rates, **arguments)
    for _ in range(2):
        with np.errstate(all="raise"):
            assert _outcome(lambda: candidate([0] * order, times)) == _outcome(
                lambda: reference([0] * order, times)
            )
    assert before == (times.tobytes(), amplitude.tobytes(), rates.tobytes())
    for name in ("amplitude", "gamma", "_rate_sum"):
        value = getattr(candidate, name)
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.setflags(write=True)


@pytest.mark.parametrize("chunk", [1, 128, 512])
@pytest.mark.parametrize(
    "amplitude,rates,times,lower",
    [
        (1e308, [1e308] * 3, [1.0] * 3, 0.0),
        (1e-308, [1e-308] * 2, [-1e308, 1e308], -1.5e308),
        (1e-308, [0.0] * 3, [1e308] * 3, -1e308),
        (1e308, [1.0] * 3, [1000.0, 1.0, 1.0], 0.0),
        (1e308, [1e-308] * 2, [1e-308] * 2, 0.0),
        (1.0, [0.0] * 2, [1e-300] * 2, 0.0),
        (np.finfo(float).max, [0.0] * 2, [2.0] * 2, 0.0),
        (complex(np.finfo(float).max, np.finfo(float).max), [0.0] * 2, [1.0] * 2, 0.0),
    ],
)
def test_ou_scaled_fallback_and_first_failure_match_frozen(
    frozen, chunk, amplitude, rates, times, lower
):
    order = len(times)
    tensor = np.full((1,) * order, amplitude)
    batch = np.repeat(np.array(times)[:, None], 129, axis=1)
    arguments = dict(t_min=lower, chunk_size=chunk)
    reference = frozen["white_cumulant"].OUWhiteCumulant(tensor, rates[0], **arguments)
    candidate = OUWhiteCumulant(tensor, rates[0], **arguments)
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error", RuntimeWarning)
        assert _outcome(lambda: candidate([0] * order, batch)) == _outcome(
            lambda: reference([0] * order, batch)
        )


def _family(module, rank):
    rng = np.random.default_rng(782 + rank)
    raw = rng.normal(size=(4, rank, rank))
    matrices = (raw + raw.swapaxes(-1, -2)) * (0.002 / np.sqrt(max(rank, 1)))
    matrices += np.eye(rank) * np.arange(1, 5)[:, None, None] / 20
    endpoints = rng.normal(size=(4, rank)) / 30
    return module.GaussianQuadraticFamily.from_factor(
        np.eye(rank), matrices, endpoints, constant=[0.0, 0.2, -0.1, 0.3]
    )


@pytest.fixture(scope="module")
def prefix_families(frozen):
    return {
        rank: (_family(frozen["quadratic"], rank), _family(quadratic, rank))
        for rank in (0, 4, 32, 128, 256)
    }


@pytest.mark.parametrize("rank", [0, 4, 32, 128, 256])
@pytest.mark.parametrize(
    "slots",
    [
        (0,),
        (1, 0),
        (2, 0, 1),
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 1, 0),
        (2, 0, 1, 0),
        (3, 0, 2, 1),
    ],
)
@pytest.mark.parametrize("divisor", [1, 24])
def test_ordered_prefix_full_call_matches_frozen(prefix_families, rank, slots, divisor):
    reference, candidate = prefix_families[rank]
    assert _outcome(lambda: candidate._mixed_statistic(slots, divisor)) == _outcome(
        lambda: reference._mixed_statistic(slots, divisor)
    )


def _native_product_count(call):
    # Observe native source-line execution without replacing ndarray arithmetic.
    path = inspect.getsourcefile(
        quadratic.GaussianQuadraticFamily._mixed_float_statistic
    )
    assert path is not None
    tree = ast.parse(Path(path).read_text())
    lines = {
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.BinOp)
        and isinstance(node.value.op, ast.MatMult)
        and any(
            isinstance(target, ast.Name) and target.id == "product"
            for target in node.targets
        )
    }
    count = 0

    def trace(frame, event, arg):
        nonlocal count
        if (
            frame.f_code.co_filename == path
            and frame.f_code.co_name == "_mixed_float_statistic"
        ):
            if event == "line" and frame.f_lineno in lines:
                count += 1
            return trace
        return None

    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        call()
    finally:
        sys.settrace(previous)
    return count


@pytest.mark.parametrize(
    "slots,products",
    [
        ((0, 0, 0, 0), 3),
        ((0, 0, 0, 1), 12),
        ((0, 0, 1, 1), 18),
        ((0, 0, 1, 2), 31),
        ((0, 1, 2, 3), 60),
    ],
)
def test_prefix_register_removes_only_identical_ordered_pairs(
    prefix_families, slots, products
):
    candidate = prefix_families[256][1]
    assert _native_product_count(lambda: candidate.mixed_cumulant(slots)) == products


def test_prefix_byte_budget_bypass_retains_original_products(
    prefix_families, monkeypatch
):
    candidate = prefix_families[256][1]
    monkeypatch.setattr(quadratic, "_MIXED_PREFIX_MAX_BYTES", 0)
    assert _native_product_count(lambda: candidate.mixed_cumulant((0, 1, 2, 3))) == 72


@pytest.mark.parametrize(
    "budget,products", [(256 * 256 * 8, 31), (256 * 256 * 8 - 1, 36)]
)
def test_prefix_exact_byte_budget_boundary(
    prefix_families, monkeypatch, budget, products
):
    candidate = prefix_families[256][1]
    monkeypatch.setattr(quadratic, "_MIXED_PREFIX_MAX_BYTES", budget)
    assert (
        _native_product_count(lambda: candidate.mixed_cumulant((0, 0, 1, 2)))
        == products
    )


@pytest.mark.parametrize("name", ["_matrices", "_endpoints"])
def test_private_non_float64_storage_keeps_original_route(name):
    candidate = _family(quadratic, 256)
    object.__setattr__(candidate, name, getattr(candidate, name).astype(np.float32))
    assert _native_product_count(lambda: candidate.mixed_cumulant((0, 1, 2, 3))) == 72


class _MatrixProtocol(np.ndarray):
    calls = 0

    def __matmul__(self, other):
        type(self).calls += 1
        return super().__matmul__(other)


def test_custom_matrix_protocol_retains_original_call_sequence(prefix_families):
    candidate = _family(quadratic, 256)
    object.__setattr__(
        candidate, "_matrices", candidate._matrices.view(_MatrixProtocol)
    )
    _MatrixProtocol.calls = 0
    candidate.mixed_cumulant((0, 1, 2, 3))
    assert _MatrixProtocol.calls == 120  # 72 ring products + 48 endpoint matvecs.


@pytest.mark.parametrize("scale", [1e-310, 1e155])
def test_prefix_first_floating_failure_rounds_the_exact_value(frozen, scale):
    matrices = np.broadcast_to(np.eye(256) * scale, (3, 256, 256)).copy()
    reference = frozen["quadratic"].GaussianQuadraticFamily.from_factor(
        np.eye(256), matrices
    )
    candidate = quadratic.GaussianQuadraticFamily.from_factor(np.eye(256), matrices)
    # The frozen reference refused at its 1e6-operation rational limit.
    assert _outcome(lambda: reference.mixed_cumulant((0, 0, 1, 2)))[0] is (
        ArithmeticError
    )
    # Each of the 24 words is tr((s I)**4) = 256 s**4 for the prepared scale s.
    prepared = float(candidate._matrices[0, 0, 0])
    assert np.array_equal(
        candidate._matrices, np.broadcast_to(np.eye(256) * prepared, (3, 256, 256))
    )
    exact = Fraction(2**3, 4) * 24 * 256 * Fraction(prepared) ** 4
    try:
        expected = ("float", struct.pack("!d", float(exact)))
    except OverflowError:
        expected = (
            OverflowError,
            ("mixed cumulant exceeds float64 range; rescale the observable",),
        )
    assert _outcome(lambda: candidate.mixed_cumulant((0, 0, 1, 2))) == expected


@pytest.mark.parametrize("slots", [(0, 0, 1, 2), (0, 1, 2, 3), (3, 0, 2, 1)])
def test_admitted_prefix_matches_independent_gauss_hermite_cumulant(slots):
    rng = np.random.default_rng(913)
    block = rng.normal(size=(4, 3, 3))
    block = block @ block.swapaxes(-1, -2) / 8
    endpoint = rng.normal(size=(4, 3)) / 3
    matrices = np.zeros((4, 256, 256))
    matrices[:, :3, :3] = block
    endpoints = np.zeros((4, 256))
    endpoints[:, :3] = endpoint
    candidate = quadratic.GaussianQuadraticFamily.from_factor(
        np.eye(256), matrices, endpoints
    )
    nodes, weights = np.polynomial.hermite.hermgauss(5)
    points = np.array(list(itertools.product(nodes * np.sqrt(2), repeat=3)))
    probability = np.array(
        [np.prod(row) for row in itertools.product(weights / np.sqrt(np.pi), repeat=3)]
    )
    values = np.einsum("ni,aij,nj->na", points, block, points) + points @ endpoint.T
    centered = values - probability @ values
    covariance = (centered.T * probability) @ centered
    a, b, c, d = slots
    expected = probability @ np.prod(centered[:, slots], axis=1)
    expected -= (
        covariance[a, b] * covariance[c, d]
        + covariance[a, c] * covariance[b, d]
        + covariance[a, d] * covariance[b, c]
    )
    assert candidate.mixed_cumulant(slots) == pytest.approx(
        expected, rel=3e-12, abs=3e-12
    )


# ---------------------------------------------------------------------------
# Mixed statistics: frozen bits, or the correctly rounded exact value
# ---------------------------------------------------------------------------


def _block_scaled_arrays(latent, seed):
    """Observables large on one block with a small tail on the other's support."""
    random = np.random.default_rng(seed)
    half = latent // 2

    def block(size):
        raw = random.normal(size=(size, size))
        return raw + raw.T

    tail = 10.0 ** -random.uniform(11, 14)
    matrices = np.zeros((3, latent, latent))
    matrices[0, :half, :half] = block(half)
    matrices[0, half:, half:] = tail * block(latent - half)
    matrices[1, :half, :half] = -0.4 * tail * block(half)
    matrices[1, half:, half:] = 1.7 * block(latent - half)
    matrices[2] = 3e-13 * block(latent)
    endpoints = random.normal(size=(3, latent)) * np.array([[0.0], [2e-7], [-1.3]])
    return matrices, endpoints


_FROZEN_BIT_SLOTS = [
    (0, 1),
    (0, 1, 1),
    (0, 0, 1),
    (0, 0, 1, 1),
    (0, 1, 1, 1),
    (2, 0, 1),
]


@pytest.mark.parametrize("latent", [6, 10, 20, 30])
def test_mixed_values_keep_frozen_bits_or_round_correctly(frozen, latent):
    # Every value the frozen 0.7.0 implementation returned is returned with
    # the same bits or as the correctly rounded exact value; a value it
    # refused is certified. The families contain queries whose float64 value
    # the rounding certificate admits but 0.7.0 sent to its exact evaluation;
    # those must keep 0.7.0's correctly rounded bits.
    flagged = 0
    for seed in range(8):
        matrices, endpoints = _block_scaled_arrays(latent, 7300 + 10 * latent + seed)
        reference = frozen["quadratic"].GaussianQuadraticFamily.from_factor(
            np.eye(latent), matrices, endpoints
        )
        candidate = quadratic.GaussianQuadraticFamily.from_factor(
            np.eye(latent), matrices, endpoints
        )
        prepared = candidate._matrices, candidate._endpoints
        assert np.array_equal(reference._matrices, prepared[0])
        assert np.array_equal(reference._endpoints, prepared[1])
        for slots in _FROZEN_BIT_SLOTS:
            expected = _outcome(lambda: reference.mixed_cumulant(slots))
            value = candidate.mixed_cumulant(slots)
            if expected == ("float", struct.pack("!d", value)):
                if expected[0] == "float":
                    approximate = float_statistic(*prepared, slots)
                    certificate = certificates.rounding_certificate(*prepared, slots, 1)
                    if (
                        certificates.admits(certificate, approximate)
                        and float(approximate) != value
                    ):
                        flagged += 1
                continue
            exact = integer_statistic(*prepared, slots)
            if expected[0] == "float":
                assert struct.pack("!d", value) == struct.pack("!d", float(exact))
            else:
                assert expected[0] is ArithmeticError
                assert_certified(value, exact)
    assert flagged


# ---------------------------------------------------------------------------
# The 0.7.0 cancellation threshold, from both sides
# ---------------------------------------------------------------------------

_UPPER, _LOWER = 7, 4


def _monomial(random, size):
    """Symmetric matrix with one nonzero per row: weights on a random involution."""
    order = random.permutation(size)
    pairs = int(random.integers(1, size // 2 + 1))
    matrix = np.zeros((size, size))
    for k in range(pairs):
        i, j = order[2 * k], order[2 * k + 1]
        matrix[i, j] = matrix[j, i] = random.normal()
    for i in order[2 * pairs :]:
        matrix[i, i] = random.normal()
    return matrix


def _threshold_arrays(seed, coupling):
    """Three observables; observable 0 is supported on the first block only.

    Observables 1 and 2 carry ``coupling`` times their first-block data and
    data of order one on the second block. Every word of a query that
    contains slot 0 passes through observable 0, so the statistic is
    ``coupling**k`` times a fixed number, with ``k`` the slots other than 0,
    while the second block keeps the Frobenius norms of order one. Every
    entry is a seeded normal draw, or ``coupling`` times one, so the arrays
    are fixed by the seed and the bits of ``coupling``. With one nonzero per
    row, every entry of a float64 product of these arrays is one rounded
    product, so their float64 contraction does not depend on the BLAS
    summation order.
    """
    random = np.random.default_rng(seed)
    latent = _UPPER + _LOWER
    matrices = np.zeros((3, latent, latent))
    endpoints = np.zeros((3, latent))
    matrices[0, :_UPPER, :_UPPER] = _monomial(random, _UPPER)
    endpoints[0, :_UPPER] = random.normal(size=_UPPER)
    for index in (1, 2):
        matrices[index, :_UPPER, :_UPPER] = coupling * _monomial(random, _UPPER)
        matrices[index, _UPPER:, _UPPER:] = _monomial(random, _LOWER)
        endpoints[index, :_UPPER] = coupling * random.normal(size=_UPPER)
        endpoints[index, _UPPER:] = random.normal(size=_LOWER)
    return matrices, endpoints


def _norm(array):
    """Frobenius norm from a correctly rounded sum of the rounded squares."""
    return math.sqrt(math.fsum(value * value for value in np.ravel(array).tolist()))


def _frobenius_bound(matrices, endpoints, slots, divisor):
    """The 0.7.0 bound: ring and endpoint-chain products of Frobenius norms.

    The sums go through ``math.fsum``, so the bits of the bound do not depend
    on the BLAS or on the summation of the builtin ``sum``, which changed in
    Python 3.12.
    """
    order = len(slots)
    matrix = {index: _norm(matrices[index]) for index in set(slots)}
    endpoint = {index: _norm(endpoints[index]) for index in set(slots)}
    ring = math.factorial(order) * 2 ** (order - 1) / order
    ring *= math.prod(matrix[index] for index in slots)
    chain = 2.0 ** (order - 3) * math.fsum(
        endpoint[word[0]]
        * endpoint[word[-1]]
        * math.prod(matrix[index] for index in word[1:-1])
        for word in itertools.permutations(slots)
    )
    return (ring + chain) / divisor


def _query_slots(query, argument):
    """Slots and divisor of a ``mixed_cumulant`` or ``joint_coefficient`` call."""
    if query == "mixed_cumulant":
        return argument, 1
    slots = tuple(index for index, power in enumerate(argument) for _ in range(power))
    return slots, math.prod(math.factorial(power) for power in argument)


_THRESHOLD = 1e-10
# Query, argument, seed, target ratio of the float64 statistic to the bound,
# and the coupling as float64 bits. Each coupling was computed once as
# 1e-10 * (ratio / s) ** (1 / k), with s the float64 statistic over
# _frobenius_bound at coupling 1e-10 and k the slots other than 0. The stored
# bits give every interpreter and BLAS the same arrays; a change of the NumPy
# random stream would move the ratio and fail its assertion. Each seed is the
# first at which all four couplings meet the preconditions the test asserts:
# no floating-point exception, the ratio within a relative 1e-3, admission by
# the certificate, and a float64 value other than the correctly rounded one.
# Python 3.10 with NumPy 2.2 and Python 3.14 with NumPy 2.4 pick the same
# seeds and couplings. No seed below 600 meets them for (0, 0, 1, 1), which
# is left out.
_THRESHOLD_CASES = [
    ("mixed_cumulant", (0, 1), 6, 2e-10, "0x1.3c1b5e7ed1c6fp-26"),
    ("mixed_cumulant", (0, 1), 6, 1.2e-10, "0x1.7b540afe95553p-27"),
    ("mixed_cumulant", (0, 1), 6, 8e-11, "0x1.f9c563fe1c718p-28"),
    ("mixed_cumulant", (0, 1), 6, 4.5e-11, "0x1.1c7f083eefffep-28"),
    ("mixed_cumulant", (0, 0, 1), 0, 2e-10, "0x1.dd8470b803dedp-28"),
    ("mixed_cumulant", (0, 0, 1), 0, 1.2e-10, "0x1.1e82aa0802528p-28"),
    ("mixed_cumulant", (0, 0, 1), 0, 8e-11, "0x1.7e038d600318ap-29"),
    ("mixed_cumulant", (0, 0, 1), 0, 4.5e-11, "0x1.adc3ff0c037bbp-30"),
    ("mixed_cumulant", (0, 1, 2), 9, 2e-10, "0x1.85d975d221428p-14"),
    ("mixed_cumulant", (0, 1, 2), 9, 1.2e-10, "0x1.2df9e0f187d4fp-14"),
    ("mixed_cumulant", (0, 1, 2), 9, 8e-11, "0x1.ed1ff8a4cbd3cp-15"),
    ("mixed_cumulant", (0, 1, 2), 9, 4.5e-11, "0x1.71d7fa7b98dedp-15"),
    ("mixed_cumulant", (0, 0, 0, 1), 4, 2e-10, "0x1.0fe646293297ap-25"),
    ("mixed_cumulant", (0, 0, 0, 1), 4, 1.2e-10, "0x1.46478764a31c6p-26"),
    ("mixed_cumulant", (0, 0, 0, 1), 4, 8e-11, "0x1.b30a09db8425dp-27"),
    ("mixed_cumulant", (0, 0, 0, 1), 4, 4.5e-11, "0x1.e96b4b16f4aa9p-28"),
    ("mixed_cumulant", (0, 1, 1, 1), 8, 2e-10, "0x1.ae6a3b87ef2b5p-8"),
    ("mixed_cumulant", (0, 1, 1, 1), 8, 1.2e-10, "0x1.6b06aaf47fa1ep-8"),
    ("mixed_cumulant", (0, 1, 1, 1), 8, 8e-11, "0x1.3d21e9082bfedp-8"),
    ("mixed_cumulant", (0, 1, 1, 1), 8, 4.5e-11, "0x1.05c97e0a5ed27p-8"),
    ("mixed_cumulant", (0, 0, 1, 2), 5, 2e-10, "0x1.48064e2f1bd8fp-12"),
    ("mixed_cumulant", (0, 0, 1, 2), 5, 1.2e-10, "0x1.fc2c6f3df62e5p-13"),
    ("mixed_cumulant", (0, 0, 1, 2), 5, 8e-11, "0x1.9eec072b63d65p-13"),
    ("mixed_cumulant", (0, 0, 1, 2), 5, 4.5e-11, "0x1.373105608ae0cp-13"),
    ("joint_coefficient", (3, 1, 0), 6, 2e-10, "0x1.5912eec8872fap-25"),
    ("joint_coefficient", (3, 1, 0), 6, 1.2e-10, "0x1.9e16b823d56c5p-26"),
    ("joint_coefficient", (3, 1, 0), 6, 8e-11, "0x1.140f256d38f2ep-26"),
    ("joint_coefficient", (3, 1, 0), 6, 4.5e-11, "0x1.36910a1ae0114p-27"),
    ("joint_coefficient", (1, 2, 1), 2, 2e-10, "0x1.1d4ffb5178e44p-8"),
    ("joint_coefficient", (1, 2, 1), 2, 1.2e-10, "0x1.e1489ce6dde4ep-9"),
    ("joint_coefficient", (1, 2, 1), 2, 8e-11, "0x1.a470a3270eddfp-9"),
    ("joint_coefficient", (1, 2, 1), 2, 4.5e-11, "0x1.5b10c8d55166ep-9"),
]


@pytest.mark.parametrize(
    "query,argument,seed,ratio,bits",
    _THRESHOLD_CASES,
    ids=[
        f"{query}-{''.join(map(str, argument))}-{ratio:g}"
        for query, argument, _, ratio, _ in _THRESHOLD_CASES
    ],
)
def test_cancellation_threshold_is_pinned_from_both_sides(
    frozen, query, argument, seed, ratio, bits
):
    # The float64 value is admitted by the rounding certificate and raises
    # no floating-point exception, so only the 0.7.0 cancellation test
    # decides between the float64 bits (above 1e-10 of the bound) and the
    # correctly rounded exact value (below it). The two differ in every case,
    # so a threshold moved past a neighbouring ratio changes a returned value.
    slots, divisor = _query_slots(query, argument)
    matrices, endpoints = _threshold_arrays(seed, float.fromhex(bits))
    with np.errstate(all="raise"):
        approximate = float_statistic(matrices, endpoints, slots, divisor)
    exact = fraction_statistic(matrices, endpoints, slots, divisor)
    bound = _frobenius_bound(matrices, endpoints, slots, divisor)
    assert abs(float(approximate)) / bound == pytest.approx(ratio, rel=1e-3, abs=0.0)
    certificate = certificates.rounding_certificate(matrices, endpoints, slots, divisor)
    assert certificates.admits(certificate, approximate)
    assert float(approximate) != float(exact)
    expected = float(approximate) if ratio > _THRESHOLD else float(exact)
    reference = frozen["quadratic"].GaussianQuadraticFamily.from_factor(
        np.eye(matrices.shape[-1]), matrices, endpoints
    )
    assert np.array_equal(reference._matrices, matrices)
    assert np.array_equal(reference._endpoints, endpoints)
    candidate = make_family(matrices, endpoints)
    outcome = ("float", struct.pack("!d", expected))
    assert _outcome(lambda: getattr(reference, query)(argument)) == outcome
    assert _outcome(lambda: getattr(candidate, query)(argument)) == outcome
