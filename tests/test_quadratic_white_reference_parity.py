"""Parity with a frozen reference implementation, and independent bounded-prefix witnesses."""

from __future__ import annotations

import ast
from fractions import Fraction
import hashlib
import inspect
import itertools
from pathlib import Path
import struct
import sys
import tarfile
import types
import warnings

import numpy as np
import pytest

from sft_wick import quadratic
from sft_wick.white_cumulant import OUWhiteCumulant


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
    path = inspect.getsourcefile(quadratic.GaussianQuadraticFamily._mixed_statistic)
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
            and frame.f_code.co_name == "_mixed_statistic"
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
