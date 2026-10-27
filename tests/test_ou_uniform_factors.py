"""Full-output and guarded-boundary controls for equal-rate OU factor scratch.

The reference disables only the new optimization flag on a separate instance.
It therefore retains the existing full-shape ordinary arithmetic and scaled
fallback; it is not an independent scientific integral oracle. The ordinary
OU validation file supplies quadrature and high-precision scientific controls.
"""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
import pickle
import warnings

import numpy as np
import pytest
from scipy.integrate import quad

from sft_wick.white_cumulant import OUWhiteCumulant


@pytest.mark.parametrize("heterogeneous", [False, True])
def test_time_factor_transcendentals_use_only_needed_component_axes(
    monkeypatch, heterogeneous
):
    gamma = [0.7, 1.0, 1.3] if heterogeneous else [0.7] * 3
    law = OUWhiteCumulant(np.ones((3, 3, 3)), gamma)
    times = np.linspace(0.4, 1.7, 15).reshape(3, 5)
    original = np.expm1
    observed = []

    def observe(array, *args, **kwargs):
        observed.append(array.shape)
        return original(array, *args, **kwargs)

    monkeypatch.setattr(np, "expm1", observe)
    result = law([0] * 3, times)
    expected_axes = (3,) * 3 if heterogeneous else (1,) * 3
    assert observed == [(5,) + expected_axes]
    assert result.shape == (5, 3, 3, 3)
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("order", [2, 3, 4, 6])
@pytest.mark.parametrize("complex_amplitude", [False, True])
def test_uniform_ordinary_matches_independent_common_time_integral(
    order, complex_amplitude
):
    amplitude = _amplitude(2, order, complex_amplitude, sparse=True)
    times = np.linspace(0.4, 1.7, order)
    lower = -0.2
    rate = 0.7
    law = OUWhiteCumulant(amplitude, rate, t_min=lower)
    outer = times[:, None]
    assert law._uniform_gamma
    assert law._ordinary_block(outer, outer.min(axis=0)) is not None
    # Independent quadrature of the original common-time convolution, with
    # all component slots retained and no compressed-factor reference helper.
    integral = quad(
        lambda tau: np.exp(-np.sum(rate * (times - tau))),
        lower,
        times.min(),
        epsabs=1e-14,
        epsrel=1e-13,
    )[0]
    np.testing.assert_allclose(
        law([0] * order, times), amplitude * integral, rtol=1e-13, atol=0.0
    )


def _uncompressed_reference(amplitude, gamma, **kwargs):
    reference = OUWhiteCumulant(amplitude, gamma, **kwargs)
    # This test-only bypass restores original scratch dimensions. It neither
    # changes immutable numerical inputs nor supplies a forged admission proof.
    object.__setattr__(reference, "_uniform_gamma", False)
    return reference


def _assert_same_bytes(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


def _amplitude(n, order, complex_amplitude, sparse=False):
    rng = np.random.default_rng(913 + n * 10 + order)
    amplitude = rng.normal(size=(n,) * order)
    if complex_amplitude:
        amplitude = amplitude + 1j * rng.normal(size=amplitude.shape)
    if sparse:
        amplitude.flat[::3] = 0.0
    if amplitude.size > 1:
        amplitude.flat[0] = complex(-0.0, -0.0) if complex_amplitude else -0.0
        amplitude.flat[1] = 0.0
    amplitude.flat[-1] = 1.25 - 0.375j if complex_amplitude else -1.25
    return amplitude


@pytest.mark.parametrize(
    "n,order", [(1, 2), (2, 2), (3, 3), (4, 2), (2, 4), (2, 6), (8, 3)]
)
@pytest.mark.parametrize("rate", [0.0, -0.0, 1.3])
@pytest.mark.parametrize("complex_amplitude", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
def test_uniform_rates_preserve_full_tensor_bytes(
    n, order, rate, complex_amplitude, sparse
):
    amplitude = _amplitude(n, order, complex_amplitude, sparse)
    if complex_amplitude:
        assert np.any(amplitude.imag != 0)
    rng = np.random.default_rng(1100 + order)
    # A strided input and mixed support exercise the existing snapshot/scatter.
    storage = np.empty((order, 34))
    times = storage[:, ::2]
    times[:] = rng.uniform(0.3, 2.0, size=times.shape)
    times += np.arange(order)[:, None] * 0.19
    times[:, 0] = 0.2
    times[:, 1] = 0.1
    candidate = OUWhiteCumulant(amplitude, rate, t_min=0.2, chunk_size=7)
    reference = _uncompressed_reference(amplitude, rate, t_min=0.2, chunk_size=7)
    assert candidate._uniform_gamma is (n > 1)
    actual = candidate(np.zeros(times.shape), times)
    assert actual.shape == (17,) + amplitude.shape
    _assert_same_bytes(actual, reference(np.zeros(times.shape), times))
    _assert_same_bytes(actual[:2], np.zeros_like(actual[:2]))


@pytest.mark.parametrize(
    "samples,chunk",
    [
        (0, 128),
        (1, 1),
        (2, 1),
        (3, 2),
        (7, 7),
        (8, 7),
        (9, 7),
        (31, 128),
        (127, 128),
        (128, 128),
        (129, 128),
        (257, 128),
    ],
)
@pytest.mark.parametrize("complex_amplitude", [False, True])
def test_short_tail_and_chunk_boundaries_keep_full_bytes(
    samples, chunk, complex_amplitude
):
    amplitude = _amplitude(3, 3, complex_amplitude)
    rng = np.random.default_rng(1200 + samples)
    times = rng.uniform(0.25, 3.0, size=(3, samples))
    candidate = OUWhiteCumulant(amplitude, 0.7, chunk_size=chunk)
    reference = _uncompressed_reference(amplitude, 0.7, chunk_size=chunk)
    _assert_same_bytes(
        candidate(np.zeros(times.shape), times),
        reference(np.zeros(times.shape), times),
    )


@pytest.mark.parametrize("complex_amplitude", [False, True])
@pytest.mark.parametrize("rate", [0.0, 0.7, np.nextafter(0.0, 1.0)])
def test_scalar_call_keeps_shape_dtype_and_signed_zero_bytes(complex_amplitude, rate):
    amplitude = _amplitude(2, 4, complex_amplitude)
    times = [0.9, 0.4, 1.3, 0.6]
    candidate = OUWhiteCumulant(amplitude, rate, t_min=-0.2)
    reference = _uncompressed_reference(amplitude, rate, t_min=-0.2)
    actual = candidate([0] * 4, times)
    assert actual.shape == amplitude.shape
    _assert_same_bytes(actual, reference([0] * 4, times))


@pytest.mark.parametrize(
    "gamma,expected",
    [
        ([0.0, 0.0], True),
        ([-0.0, -0.0], True),
        ([0.0, -0.0], False),
        ([0.7, 0.7], True),
        ([0.7, np.nextafter(0.7, 1.0)], False),
        ([0.7, 1.3], False),
        ([np.nextafter(0.0, 1.0)] * 2, True),
        ([1e308, 1e308], True),
    ],
)
def test_exact_rate_bits_admit_only_identical_components(gamma, expected):
    amplitude = _amplitude(2, 3, True)
    candidate = OUWhiteCumulant(amplitude, gamma)
    reference = _uncompressed_reference(amplitude, gamma)
    assert candidate._uniform_gamma is expected
    times = np.array([[0.4, 0.8, 1.1], [0.9, 0.5, 0.4], [0.6, 0.7, 1.0]])
    _assert_same_bytes(
        candidate(np.zeros(times.shape), times),
        reference(np.zeros(times.shape), times),
    )


@pytest.mark.parametrize(
    "magnitude,rate,times,lower",
    [
        (1e308, 1.0, [1000.0, 1.0], 0.0),
        (1e308, 1e308, [1.0, 1.0, 1.0], 0.0),
        (1e-308, 1e-308, [-1e308, 1e308], -1.5e308),
        (0.5, 0.0, [np.finfo(float).max] * 2, -np.finfo(float).max),
        (np.finfo(float).max, 0.0, [1.0, 1.0], 0.0),
        (1e308, 1e308, [1e-307, 4.5e-307], 0.0),
    ],
)
@pytest.mark.parametrize("complex_amplitude", [False, True])
def test_scaled_denial_preserves_extreme_result_and_errors(
    magnitude, rate, times, lower, complex_amplitude
):
    order = len(times)
    amplitude = np.full((2,) * order, magnitude)
    if complex_amplitude:
        amplitude = amplitude.astype(complex)
        amplitude.imag[:] = -magnitude / 2
    candidate = OUWhiteCumulant(amplitude, rate, t_min=lower)
    reference = _uncompressed_reference(amplitude, rate, t_min=lower)
    outer = np.array(times)[:, None]
    assert candidate._ordinary_block(outer, outer.min(axis=0)) is None
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(all="raise"):
            actual = candidate([0] * order, times)
            expected = reference([0] * order, times)
    assert np.all(np.isfinite(actual))
    _assert_same_bytes(actual, expected)


@pytest.mark.parametrize("complex_amplitude", [False, True])
def test_unused_component_overflow_is_still_reported(complex_amplitude):
    amplitude = np.zeros((2,) * 3)
    if complex_amplitude:
        amplitude = amplitude.astype(complex)
        amplitude.flat[0] = 0.125 + 0.25j
        amplitude.flat[-1] = complex(0.0, np.finfo(float).max)
    else:
        amplitude.flat[0] = 0.125
        amplitude.flat[-1] = np.finfo(float).max
    for factory in (OUWhiteCumulant, _uncompressed_reference):
        law = factory(amplitude, 0.0)
        with pytest.raises(OverflowError, match="float64 range"):
            law([0] * 3, [2.0] * 3)


def test_whole_output_budget_precedes_blocks(monkeypatch):
    amplitude = np.ones((2, 2))
    law = OUWhiteCumulant(amplitude, 0.7, chunk_size=1, max_output_bytes=64)

    def unexpected_block(self, times):
        raise AssertionError("whole batch budget must be checked before work blocks")

    monkeypatch.setattr(OUWhiteCumulant, "_block", unexpected_block)
    with pytest.raises(MemoryError, match="batch exceeds max_output_bytes"):
        law(np.zeros((2, 3)), np.ones((2, 3)))


def test_uniform_metadata_snapshots_copy_pickle_and_result_ownership():
    amplitude = _amplitude(2, 3, True)
    gamma = np.full(2, 0.7)
    times = np.array([[0.4, 0.8], [0.9, 0.5], [0.6, 0.7]])
    law = OUWhiteCumulant(
        amplitude, gamma, t_min=-0.2, chunk_size=1, max_output_bytes=4096
    )
    expected = law(np.zeros(times.shape), times)
    amplitude[:] = 100.0
    gamma[1] = 2.0
    assert copy.copy(law) is law
    assert copy.deepcopy(law) is law
    for target in (law, pickle.loads(pickle.dumps(law))):
        assert target._uniform_gamma is True
        assert target.chunk_size == 1 and target.max_output_bytes == 4096
        actual = target(np.zeros(times.shape), times)
        _assert_same_bytes(actual, expected)
        assert not np.shares_memory(actual, target.amplitude)
        assert not np.shares_memory(actual, times)
        for values in (target.amplitude, target.gamma, target._rate_sum):
            with pytest.raises(ValueError):
                values.setflags(write=True)
        with pytest.raises(FrozenInstanceError):
            target._uniform_gamma = False
        actual[:] = 0.0
        _assert_same_bytes(target(np.zeros(times.shape), times), expected)
