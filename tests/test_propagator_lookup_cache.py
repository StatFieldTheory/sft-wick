"""Exact key, ownership, budget and lifecycle contracts for lookup reuse."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from sft_wick._propagator_lookup import (
    _PropagatorLookupCache,
    validate_cache_options,
    wrap_propagator_cache,
)


class _SpyCache:
    def __init__(self, output=None):
        self.calls = 0
        self.output = output
        self.value = 2.0
        self._private_flag = object()
        self.scalar_calls = 0

    def R_time_batch(self, *args, **kwargs):
        self.calls += 1
        if self.output is not None:
            return self.output
        return np.array([self.value])

    def R_product(self, *args):
        self.scalar_calls += 1
        return self.scalar_calls

    def clear_cache(self):
        self.value += 1


class _FullCache(_SpyCache):
    def R_matrix_batch(self, *args, **kwargs):
        return self.R_time_batch(*args, **kwargs)

    def C_at_batch(self, *args, **kwargs):
        return self.R_time_batch(*args, **kwargs)

    def C_diagonal_batch(self, *args, **kwargs):
        return self.R_time_batch(*args, **kwargs)

    def C_matrix_batch(self, *args, **kwargs):
        return self.R_time_batch(*args, **kwargs)

    def precompute_C_table(self, *args, **kwargs):
        self.value += 1

    precompute_C_table_translation = precompute_C_table
    precompute_C_table_rotation = precompute_C_table
    precompute_C_table_general = precompute_C_table


class _ArrayProtocol:
    def __init__(self):
        self.conversions = 0

    def __array__(self, *args, **kwargs):
        self.conversions += 1
        raise AssertionError("unsupported input must not be converted")


class _ArraySubclass(np.ndarray):
    def __array__(self, *args, **kwargs):
        raise AssertionError("subclass must not be converted")


class _ScalarSubclass(float):
    def __float__(self):
        raise AssertionError("scalar subclass must not be converted")


class _NumpyScalarSubclass(np.float64):
    def __float__(self):
        raise AssertionError("numpy subclass must not be converted")


class _SerializedSource(_SpyCache):
    def __init__(self):
        super().__init__()
        self.numeric_memo = {"drop": np.ones(10)}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["numeric_memo"] = {}
        return state


def _wrap(cache, budget=100_000):
    return wrap_propagator_cache(cache, budget, "fixed_model")


@pytest.mark.parametrize("budget", [None, 0])
@pytest.mark.parametrize("contract", [None, "fixed_model"])
def test_disabled_is_identity_and_invokes_no_callback(budget, contract):
    source = _SpyCache()
    assert validate_cache_options(budget, contract) == 0
    assert wrap_propagator_cache(source, budget, contract) is source
    assert source.calls == 0


@pytest.mark.parametrize("budget", [True, False, 1.0, "1", np.int64(1), object()])
def test_budget_rejects_non_native_integer_before_lookup(budget):
    source = _SpyCache()
    with pytest.raises(TypeError, match="propagator_cache_bytes"):
        wrap_propagator_cache(source, budget, "fixed_model")
    assert source.calls == 0


@pytest.mark.parametrize(
    "budget,contract,exception",
    [
        (-1, "fixed_model", ValueError),
        (1, None, ValueError),
        (1, "auto", ValueError),
        (0, "auto", ValueError),
        (1, object(), TypeError),
        (1, np.array([1]), TypeError),
    ],
)
def test_invalid_contract_or_negative_budget_fails_upfront(budget, contract, exception):
    source = _SpyCache()
    with pytest.raises(exception):
        wrap_propagator_cache(source, budget, contract)
    assert source.calls == 0


def test_validation_does_not_probe_custom_contract_equality():
    class InvalidContract:
        def __eq__(self, other):
            raise AssertionError("option validation must not invoke arbitrary equality")

    with pytest.raises(TypeError, match="propagator_cache_contract"):
        validate_cache_options(1, InvalidContract())


def test_all_five_methods_have_independent_keys_and_hits():
    source = _FullCache()
    cache = _wrap(source)
    for name in (
        "R_time_batch",
        "R_matrix_batch",
        "C_at_batch",
        "C_diagonal_batch",
        "C_matrix_batch",
    ):
        np.testing.assert_array_equal(getattr(cache, name)(1.0), [2.0])
        np.testing.assert_array_equal(getattr(cache, name)(1.0), [2.0])
    assert source.calls == 5
    assert cache._hits == 5


@pytest.mark.parametrize(
    "first,second",
    [
        (np.array([1.0], dtype="f8"), np.array([1.0], dtype="f4")),
        (np.array([1.0], dtype="<f8"), np.array([1.0], dtype=">f8")),
        (np.array([1.0, 2.0]), np.array([[1.0, 2.0]])),
        (np.array([1.0, 2.0]), np.array([2.0, 1.0])),
        (np.array([0.0]), np.array([-0.0])),
        (np.array([1]), np.array([True])),
        (1.0, np.float64(1.0)),
        (np.float64(1.0), np.array(1.0)),
        (1, 1.0),
        (False, 0),
        (0.0, -0.0),
        (complex(0.0, 0.0), complex(0.0, -0.0)),
    ],
)
def test_exact_input_type_dtype_shape_contents_and_signed_zero(first, second):
    source = _SpyCache()
    cache = _wrap(source)
    cache.R_time_batch(first)
    cache.R_time_batch(second)
    cache.R_time_batch(first)
    cache.R_time_batch(second)
    assert source.calls == 2


def test_equal_norm_spatial_vectors_and_leg_order_are_not_identified():
    source = _SpyCache()
    cache = _wrap(source)
    x, y = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    cache.R_time_batch(x, y)
    cache.R_time_batch(y, x)
    cache.R_time_batch(x, y)
    assert source.calls == 2


def test_ordered_keywords_and_positional_binding_are_distinct():
    source = _SpyCache()
    cache = _wrap(source)
    cache.R_time_batch(t1=1.0, t2=2.0)
    cache.R_time_batch(t2=2.0, t1=1.0)
    cache.R_time_batch(1.0, 2.0)
    cache.R_time_batch(t1=1.0, t2=2.0)
    assert source.calls == 3


def test_array_layout_and_writeability_are_conservative_key_guards():
    source = _SpyCache()
    cache = _wrap(source)
    normal = np.array([1.0, 2.0])
    strided = np.array([1.0, 0.0, 2.0, 0.0])[::2]
    read_only = normal.copy()
    read_only.flags.writeable = False
    for argument in (normal, strided, read_only):
        cache.R_time_batch(argument)
        cache.R_time_batch(argument)
    assert source.calls == 3


def test_array_contents_are_read_each_call_and_saved_keys_own_bytes():
    source = _SpyCache()
    cache = _wrap(source)
    argument = np.array([1.0])
    cache.R_time_batch(argument)
    argument[0] = 9.0
    cache.R_time_batch(argument)
    argument[0] = 1.0
    cache.R_time_batch(argument)
    assert source.calls == 2


@pytest.mark.parametrize(
    "argument",
    [
        [1.0],
        (1.0,),
        None,
        "1.0",
        np.array([object()]),
        np.array([1.0]).view(_ArraySubclass),
        _ScalarSubclass(1.0),
        _NumpyScalarSubclass(1.0),
        np.array([1.0], dtype=np.dtype("f8", metadata={"tag": "custom"})),
        np.datetime64("2026-10-12"),
    ],
)
def test_unsupported_input_bypasses_without_protocol_coercion(argument):
    source = _SpyCache()
    cache = _wrap(source)
    cache.R_time_batch(argument)
    cache.R_time_batch(argument)
    assert source.calls == 2
    assert cache._retained_bytes == 0


def test_arbitrary_array_protocol_is_never_probed():
    source = _SpyCache()
    cache = _wrap(source)
    argument = _ArrayProtocol()
    cache.R_time_batch(argument)
    cache.R_time_batch(argument)
    assert argument.conversions == 0
    assert source.calls == 2


@pytest.mark.parametrize(
    "argument",
    [
        float("nan"),
        float("inf"),
        complex(1, float("inf")),
        np.float64("nan"),
        np.array([1.0, float("inf")]),
    ],
)
def test_nonfinite_input_bypasses_and_underlying_sees_it(argument):
    source = _SpyCache()
    cache = _wrap(source)
    cache.R_time_batch(argument)
    cache.R_time_batch(argument)
    assert source.calls == 2
    assert not cache._entries


@pytest.mark.parametrize(
    "output",
    [
        float("nan"),
        float("inf"),
        np.array([float("inf")]),
        np.array([1.0 + float("nan") * 1j]),
        np.array([object()]),
        np.array([1.0]).view(_ArraySubclass),
        _ArrayProtocol(),
        [1.0],
    ],
)
def test_nonfinite_and_unsupported_output_is_returned_once_per_call(output):
    source = _SpyCache(output)
    cache = _wrap(source)
    assert cache.R_time_batch(1.0) is output
    assert cache.R_time_batch(1.0) is output
    assert source.calls == 2
    assert cache._retained_bytes == 0
    if isinstance(output, _ArrayProtocol):
        assert output.conversions == 0


@pytest.mark.parametrize(
    "output",
    [True, 2, -(10**200), 0.0, -0.0, complex(-0.0, -0.0), np.float32(-0.0)],
)
def test_native_numeric_scalar_outputs_preserve_type_and_signed_zero(output):
    source = _SpyCache(output)
    cache = _wrap(source)
    miss = cache.R_time_batch(1.0)
    hit = cache.R_time_batch(1.0)
    assert type(hit) is type(miss)
    assert pickle.dumps(hit) == pickle.dumps(miss)
    assert source.calls == 1


@pytest.mark.parametrize("dtype", ["f4", "f8", "c16", "i8", "bool"])
def test_output_dtype_shape_signed_zero_and_boolean_values_preserved(dtype):
    output = np.array([[0, -0.0], [1, 2]], dtype=dtype)
    source = _SpyCache(output)
    cache = _wrap(source)
    miss = cache.R_time_batch(1.0)
    hit = cache.R_time_batch(1.0)
    assert miss is output
    assert hit.shape == output.shape
    assert hit.dtype == output.dtype
    assert hit.tobytes() == output.tobytes()
    assert not np.shares_memory(hit, output)
    assert source.calls == 1


def test_miss_and_hit_mutations_cannot_poison_owned_storage():
    output = np.array([3.0, -0.0])
    source = _SpyCache(output)
    cache = _wrap(source)
    miss = cache.R_time_batch(1.0)
    miss[:] = 100.0
    hit = cache.R_time_batch(1.0)
    assert hit.tobytes() == np.array([3.0, -0.0]).tobytes()
    hit[:] = -100.0
    next_hit = cache.R_time_batch(1.0)
    assert next_hit.tobytes() == np.array([3.0, -0.0]).tobytes()
    assert not np.shares_memory(next_hit, hit)
    assert source.calls == 1


def test_native_view_output_is_copied_and_its_source_cannot_poison_storage():
    buffer = np.array([1.0, 2.0, 3.0, 4.0])
    output = buffer.reshape(2, 2).T
    source = _SpyCache(output)
    cache = _wrap(source)
    assert cache.R_time_batch(1.0) is output
    buffer[:] = 0.0
    hit = cache.R_time_batch(1.0)
    np.testing.assert_array_equal(hit, [[1.0, 3.0], [2.0, 4.0]])
    assert hit.strides == output.strides
    assert not np.shares_memory(hit, buffer)
    assert source.calls == 1


@pytest.mark.parametrize("view", ["holes", "negative", "broadcast"])
def test_output_layout_that_a_copy_cannot_preserve_bypasses(view):
    original = np.arange(4.0)
    if view == "holes":
        output = original[::2]
    elif view == "negative":
        output = original[::-1]
    else:
        output = np.broadcast_to(original[:1], (4,))
    source = _SpyCache(output)
    cache = _wrap(source)
    assert cache.R_time_batch(1.0) is output
    assert cache.R_time_batch(1.0) is output
    assert source.calls == 2
    assert not cache._entries


def test_readonly_output_remains_readonly_on_every_hit():
    output = np.array([1.0])
    output.flags.writeable = False
    source = _SpyCache(output)
    cache = _wrap(source)
    assert cache.R_time_batch(1.0) is output
    for _ in range(2):
        hit = cache.R_time_batch(1.0)
        assert not hit.flags.writeable
        assert not np.shares_memory(hit, output)
        with pytest.raises(ValueError):
            hit[0] = 9.0
    assert source.calls == 1


def test_missing_capabilities_and_private_flags_are_delegated_exactly():
    source = _SpyCache()
    cache = _wrap(source)
    assert cache._private_flag is source._private_flag
    assert not hasattr(cache, "C_matrix_batch")
    assert not hasattr(cache, "_c_splines")
    assert not hasattr(cache, "precompute_C_table")
    with pytest.raises(AttributeError):
        cache.C_matrix_batch
    assert cache.R_product.__self__ is source
    assert cache.R_product() == 1
    assert cache.R_product() == 2
    assert source.calls == 0


def test_noncallable_batch_capability_is_not_promoted():
    source = _SpyCache()
    source.C_matrix_batch = False
    cache = _wrap(source)
    assert cache.C_matrix_batch is False


def test_exceptions_are_never_cached_or_hidden():
    class Failing(_SpyCache):
        def R_time_batch(self, *args, **kwargs):
            self.calls += 1
            raise ValueError("causal fixture failure")

    source = Failing()
    cache = _wrap(source)
    for _ in range(2):
        with pytest.raises(ValueError, match="causal fixture failure"):
            cache.R_time_batch(1.0)
    assert source.calls == 2
    assert not cache._entries


def test_original_causal_mask_and_callback_errors_are_not_preprobed():
    class Causal(_SpyCache):
        def R_time_batch(self, left, right):
            self.calls += 1
            result = np.zeros(left.shape)
            for index, (t1, t2) in enumerate(zip(left, right)):
                if t1 > t2:
                    if t1 == 3.0:
                        raise RuntimeError("causal callback raises")
                    result[index] = t1 - t2
            return result

    source = Causal()
    cache = _wrap(source)
    left, right = np.array([3.0, 1.0, 2.0]), np.array([4.0, 1.0, 0.0])
    np.testing.assert_array_equal(cache.R_time_batch(left, right), [0, 0, 2])
    np.testing.assert_array_equal(cache.R_time_batch(left, right), [0, 0, 2])
    assert source.calls == 1
    for _ in range(2):
        with pytest.raises(RuntimeError, match="causal callback raises"):
            cache.R_time_batch(np.array([3.0]), np.array([0.0]))
    assert source.calls == 3


def test_budget_bounds_retained_raw_keys_and_outputs_after_lru_eviction():
    source = _SpyCache()
    cache = _wrap(source, 70)
    cache.R_time_batch(1.0)
    cache.R_time_batch(2.0)
    cache.R_time_batch(1.0)  # 1 is now the hot entry.
    cache.R_time_batch(3.0)  # 2 is evicted.
    cache.R_time_batch(1.0)
    cache.R_time_batch(2.0)
    assert source.calls == 4
    assert cache._retained_bytes <= 70
    assert len(cache._entries) == 2


def test_oversized_output_does_not_evict_existing_useful_entries():
    source = _SpyCache()
    cache = _wrap(source, 70)
    cache.R_time_batch(1.0)
    before_bytes = cache._retained_bytes
    source.output = np.ones(100)
    cache.R_time_batch(2.0)
    assert cache._retained_bytes == before_bytes
    source.output = None
    cache.R_time_batch(1.0)
    assert source.calls == 2


def test_oversized_input_bypasses_before_finite_scan_or_key_byte_copy(monkeypatch):
    source = _SpyCache()
    cache = _wrap(source, 70)
    huge = np.ones(1000)

    def forbidden(*args, **kwargs):
        raise AssertionError("oversized argument must bypass before finite scan")

    monkeypatch.setattr(np, "isfinite", forbidden)
    cache.R_time_batch(huge)
    cache.R_time_batch(huge)
    assert source.calls == 2
    assert cache._retained_bytes == 0


def test_key_bytes_are_charged_even_when_output_is_empty():
    source = _SpyCache(np.empty(0))
    cache = _wrap(source, 70)
    cache.R_time_batch(np.ones(8))
    cache.R_time_batch(np.ones(8))
    assert source.calls == 2
    assert cache._retained_bytes == 0


def test_oversized_keyword_name_bypasses_without_key_payload_copy(monkeypatch):
    import sft_wick._propagator_lookup as lookup

    source = _SpyCache()
    cache = _wrap(source, 70)

    def forbidden(*args, **kwargs):
        raise AssertionError("oversized keyword must bypass before payload encoding")

    monkeypatch.setattr(lookup, "_argument_bytes", forbidden)
    cache.R_time_batch(**{"long" * 100: 1.0})
    assert source.calls == 1
    assert not cache._entries


def test_entry_count_cap_bounds_empty_output_metadata():
    source = _SpyCache(np.empty(0))
    cache = _wrap(source, 1_000_000)
    for index in range(4100):
        cache.R_time_batch(index)
        assert cache._retained_bytes <= 1_000_000
        assert len(cache._entries) <= 4096
    assert len(cache._entries) == 4096
    cache.R_time_batch(0)
    assert source.calls == 4101


@pytest.mark.parametrize(
    "operation",
    [
        "clear_cache",
        "precompute_C_table",
        "precompute_C_table_translation",
        "precompute_C_table_rotation",
        "precompute_C_table_general",
    ],
)
def test_clear_and_table_build_hooks_invalidate_before_reconfiguration(operation):
    source = _FullCache()
    cache = _wrap(source)
    np.testing.assert_array_equal(cache.R_time_batch(1.0), [2.0])
    getattr(cache, operation)()
    assert not cache._entries
    assert cache._retained_bytes == 0
    np.testing.assert_array_equal(cache.R_time_batch(1.0), [3.0])
    assert source.calls == 2


def test_failing_clear_invalidates_even_if_underlying_mutates_then_raises():
    class FailingClear(_SpyCache):
        def clear_cache(self):
            self.value = 7.0
            raise RuntimeError("clear failed")

    source = FailingClear()
    cache = _wrap(source)
    cache.R_time_batch(1.0)
    with pytest.raises(RuntimeError, match="clear failed"):
        cache.clear_cache()
    assert not cache._entries
    np.testing.assert_array_equal(cache.R_time_batch(1.0), [7.0])


def test_reentrant_clear_does_not_reinsert_the_inflight_result():
    class Reentrant(_SpyCache):
        def R_time_batch(self, *args, **kwargs):
            self.calls += 1
            wrapper.clear_cache()
            return np.array([self.value])

    source = Reentrant()
    wrapper = _wrap(source)
    wrapper.R_time_batch(1.0)
    assert not wrapper._entries
    assert wrapper._retained_bytes == 0


def test_close_frees_only_local_bank_and_future_calls_delegate():
    source = _SpyCache()
    cache = _wrap(source)
    cache.R_time_batch(1.0)
    cache.close()
    cache.close()
    assert source.value == 2.0
    assert cache._retained_bytes == 0
    assert not cache._entries
    cache.R_time_batch(1.0)
    cache.R_time_batch(1.0)
    assert source.calls == 3
    assert not cache._entries


def test_matching_active_wrapper_is_reused_and_incompatible_nesting_fails():
    source = _SpyCache()
    cache = _wrap(source)
    assert _wrap(cache) is cache
    with pytest.raises(ValueError, match="active scope"):
        _wrap(cache, 200_000)
    cache.close()
    fresh = _wrap(cache)
    assert fresh is not cache
    assert fresh._source is source
    assert not isinstance(fresh._source, _PropagatorLookupCache)
    assert not fresh._closed


def test_separate_invocations_and_models_cannot_share_numeric_entries():
    source = _SpyCache()
    first = _wrap(source)
    np.testing.assert_array_equal(first.R_time_batch(1.0), [2.0])
    first.close()
    source.value = 9.0
    second = _wrap(source)
    np.testing.assert_array_equal(second.R_time_batch(1.0), [9.0])
    other_model = _wrap(_SpyCache())
    np.testing.assert_array_equal(other_model.R_time_batch(1.0), [2.0])
    assert source.calls == 2


def test_pickle_drops_bank_and_counters_preserves_underlying_state():
    source = _SpyCache()
    cache = _wrap(source)
    cache.R_time_batch(1.0)
    cache.R_time_batch(1.0)
    restored = pickle.loads(pickle.dumps(cache))
    assert not restored._entries
    assert restored._retained_bytes == 0
    assert restored._hits == restored._misses == 0
    assert restored._source.calls == 1
    np.testing.assert_array_equal(restored.R_time_batch(1.0), [2.0])
    assert restored._source.calls == 2
    assert cache._source.calls == 1
    assert len(cache._entries) == 1


def test_pickle_preserves_source_serialization_policy_and_closed_scope():
    source = _SpyCache()
    cache = _wrap(source)
    cache.close()
    restored = pickle.loads(pickle.dumps(cache))
    assert restored._closed
    restored.R_time_batch(1.0)
    restored.R_time_batch(1.0)
    assert restored._source.calls == 2
    assert not restored._entries


def test_source_owns_its_serialization_policy_independently_of_lookup_bank():
    source = _SerializedSource()
    cache = _wrap(source)
    cache.R_time_batch(1.0)
    restored = pickle.loads(pickle.dumps(cache))
    assert restored._source.numeric_memo == {}
    assert not restored._entries
    assert "drop" in source.numeric_memo


def test_attribute_lookup_before_unpickle_state_is_restored_does_not_recurse():
    empty = _PropagatorLookupCache.__new__(_PropagatorLookupCache)
    for name in ("_source", "__setstate__missing", "C_matrix_batch"):
        with pytest.raises(AttributeError):
            getattr(empty, name)


@pytest.mark.parametrize(
    "left,right",
    [
        (np.float64, np.longdouble),
        (np.complex128, np.clongdouble),
        (np.int64, np.longlong),
    ],
)
@pytest.mark.parametrize("as_array", [False, True])
def test_platform_native_dtype_aliases_keep_separate_keys(left, right, as_array):
    class DtypeSensitive:
        def __init__(self):
            self.calls = 0

        def R_time_batch(self, value):
            self.calls += 1
            return np.array([float(value.dtype.num)])

    source = DtypeSensitive()
    bank = wrap_propagator_cache(source, 4096, "fixed_model")
    first, second = left(1), right(1)
    if as_array:
        first, second = np.array([first]), np.array([second])
    a, b = bank.R_time_batch(first), bank.R_time_batch(second)
    np.testing.assert_array_equal(a, [float(first.dtype.num)])
    np.testing.assert_array_equal(b, [float(second.dtype.num)])
    np.testing.assert_array_equal(bank.R_time_batch(first), a)
    np.testing.assert_array_equal(bank.R_time_batch(second), b)
    assert source.calls == (1 if first.dtype.num == second.dtype.num else 2)
