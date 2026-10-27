"""Bounded, explicitly promised reuse of batched propagator lookups.

The bank is local to an owning integration call.  ``fixed_model`` promises
deterministic values, unchanged model/table semantics and callbacks that do
not depend on how often they are invoked.  This is a caller contract, not a
purity test.  Stored and hit arrays are independent copies, so callers must
not depend on shared output-buffer identity.  The byte budget covers raw key
bytes and stored numeric payloads; Python metadata, transient copies and the
underlying propagator tables are outside that budget.
"""

from __future__ import annotations

from collections import OrderedDict
import math
import struct
from typing import Any

import numpy as np


_BATCH_METHODS = frozenset(
    {
        "R_time_batch",
        "R_matrix_batch",
        "C_at_batch",
        "C_diagonal_batch",
        "C_matrix_batch",
    }
)
_RECONFIGURE_METHODS = frozenset(
    {
        "clear_cache",
        "precompute_C_table",
        "precompute_C_table_translation",
        "precompute_C_table_rotation",
        "precompute_C_table_general",
    }
)
_MAX_ENTRIES = 4096
_NUMPY_SCALAR_TYPES = frozenset(
    scalar_type
    for scalar_type in np.sctypeDict.values()
    if np.dtype(scalar_type).kind in "biufc"
)


def validate_cache_options(
    propagator_cache_bytes: int | None = None,
    propagator_cache_contract: str | None = None,
) -> int:
    """Validate options without inspecting the cache or invoking callbacks.

    Return the normalized raw-byte budget.  ``None`` and zero disable reuse;
    a positive budget requires the explicit ``'fixed_model'`` contract.
    """
    if propagator_cache_bytes is None:
        budget = 0
    elif type(propagator_cache_bytes) is not int:
        raise TypeError("propagator_cache_bytes must be an integer or None")
    else:
        budget = propagator_cache_bytes
    if budget < 0:
        raise ValueError("propagator_cache_bytes must be nonnegative")
    if (
        propagator_cache_contract is not None
        and type(propagator_cache_contract) is not str
    ):
        raise TypeError("propagator_cache_contract must be a string or None")
    if propagator_cache_contract not in (None, "fixed_model"):
        raise ValueError("propagator_cache_contract must be None or 'fixed_model'")
    if budget and propagator_cache_contract != "fixed_model":
        raise ValueError(
            "positive propagator_cache_bytes requires "
            "propagator_cache_contract='fixed_model'"
        )
    return budget


def _int_bytes(value: int) -> bytes:
    # A sign byte keeps positive/negative integers separate without coercion.
    magnitude = abs(value)
    size = (magnitude.bit_length() + 7) // 8
    return bytes((value < 0,)) + magnitude.to_bytes(size, "big")


def _numeric_size(value: Any) -> int | None:
    """Preflight size of a supported numeric value, without copying arrays."""
    value_type = type(value)
    if value_type is np.ndarray:
        if value.dtype.kind not in "biufc" or value.dtype.metadata is not None:
            return None
        return value.nbytes
    if value_type in _NUMPY_SCALAR_TYPES:
        return value.dtype.itemsize
    if value_type is bool:
        return 1
    if value_type is int:
        return 1 + (value.bit_length() + 7) // 8
    if value_type is float:
        return 8
    if value_type is complex:
        return 16
    return None


def _finite(value: Any) -> bool:
    value_type = type(value)
    if value_type is int or value_type is bool:
        return True
    if value_type is float:
        return math.isfinite(value)
    if value_type is complex:
        return math.isfinite(value.real) and math.isfinite(value.imag)
    return bool(np.isfinite(value).all())


def _argument_header(value: Any) -> bytes:
    """Encode exact type, dtype and shape; called only after admission."""
    value_type = type(value)
    if value_type is np.ndarray:
        dtype = value.dtype.str.encode("ascii")
        shape = b"".join(struct.pack("!Q", dimension) for dimension in value.shape)
        strides = b"".join(struct.pack("!q", stride) for stride in value.strides)
        flags = bytes(
            (
                value.flags.c_contiguous,
                value.flags.f_contiguous,
                value.flags.owndata,
                value.flags.writeable,
                value.flags.aligned,
                value.flags.writebackifcopy,
            )
        )
        return (
            b"a"
            + dtype
            + struct.pack("!I", value.dtype.num)
            + b":"
            + struct.pack("!Q", value.ndim)
            + shape
            + strides
            + flags
        )
    if value_type in _NUMPY_SCALAR_TYPES:
        # dtype.str can alias distinct native types on some platforms:
        # float64/longdouble, complex128/clongdouble and int64/longlong.
        return (
            b"n"
            + value.dtype.str.encode("ascii")
            + struct.pack("!I", value.dtype.num)
            + b":"
        )
    return {bool: b"b", int: b"i", float: b"f", complex: b"c"}[value_type]


def _argument_bytes(value: Any) -> bytes:
    value_type = type(value)
    if value_type is np.ndarray or value_type in _NUMPY_SCALAR_TYPES:
        return value.tobytes(order="C")
    if value_type is bool:
        return bytes((value,))
    if value_type is int:
        return _int_bytes(value)
    if value_type is float:
        return struct.pack("!d", value)
    return struct.pack("!dd", value.real, value.imag)


def _make_key(
    method: str, args: tuple[Any, ...], kwargs: dict[str, Any], budget: int
) -> tuple[tuple[Any, ...], int] | None:
    """Check total raw size before making any retained argument-byte copies."""
    # Keep boundaries structural, rather than concatenating ambiguous bytes.
    named = tuple(kwargs.items())
    values = args + tuple(value for _, value in named)
    headers: list[bytes] = []
    raw_size = len(method.encode("ascii"))
    if raw_size > budget:
        return None
    for name, _ in named:
        # Python keyword names are strings; subclasses have arbitrary methods.
        if type(name) is not str:
            return None
        if len(name) > budget - raw_size:
            return None
        raw_size += len(name.encode("utf-8"))
        if raw_size > budget:
            return None
    for value in values:
        numeric_size = _numeric_size(value)
        if numeric_size is None or numeric_size > budget:
            return None
        header = _argument_header(value)
        raw_size += len(header) + numeric_size
        if raw_size > budget:
            return None
        headers.append(header)
    if not all(_finite(value) for value in values):
        return None
    encoded = tuple(
        (header, _argument_bytes(value)) for header, value in zip(headers, values)
    )
    key = (method, len(args), tuple(name for name, _ in named), encoded)
    return key, raw_size


def _copy_payload(value: Any) -> Any:
    if type(value) is np.ndarray:
        copied = value.copy(order="K")
        if not value.flags.writeable:
            copied.flags.writeable = False
        return copied
    # Admitted scalar types are immutable.
    return value


class _PropagatorLookupCache:
    """Composition proxy with a bounded LRU and unchanged duck capabilities.

    Only batch methods present on the wrapped object are intercepted.  All
    other attributes, including bound scalar methods, retain their original
    dispatch.  Query owners call :meth:`close` at the end of the scope.
    """

    def __init__(self, cache: Any, budget: int, contract: str) -> None:
        self._source = cache
        self._budget = validate_cache_options(budget, contract)
        self._contract = contract
        self._entries: OrderedDict[tuple[Any, ...], tuple[Any, int]] = OrderedDict()
        self._retained_bytes = 0
        self._hits = 0
        self._misses = 0
        self._closed = False
        self._generation = 0

    def __getattr__(self, name: str) -> Any:
        # Pickle probes attributes before restoring state.  Ordinary delegated
        # getattr here would recursively enter this method while _source is absent.
        try:
            source = object.__getattribute__(self, "_source")
        except AttributeError:
            raise AttributeError(name) from None
        attribute = getattr(source, name)
        if name in _BATCH_METHODS and callable(attribute):

            def lookup(*args: Any, **kwargs: Any) -> Any:
                return self._lookup(name, attribute, args, kwargs)

            return lookup
        if name in _RECONFIGURE_METHODS and callable(attribute):

            def reconfigure(*args: Any, **kwargs: Any) -> Any:
                # An underlying failure can still mutate its state.  Invalidate
                # before calling, so no stale entry survives either outcome.
                self._clear_entries()
                return attribute(*args, **kwargs)

            return reconfigure
        return attribute

    def _lookup(
        self, name: str, method: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        if self._closed:
            return method(*args, **kwargs)
        candidate = _make_key(name, args, kwargs, self._budget)
        if candidate is None:
            return method(*args, **kwargs)
        key, key_size = candidate
        entry = self._entries.get(key)
        if entry is not None:
            self._hits += 1
            self._entries.move_to_end(key)
            return _copy_payload(entry[0])
        self._misses += 1
        generation = self._generation
        output = method(*args, **kwargs)
        if generation != self._generation or self._closed:
            return output
        output_size = _numeric_size(output)
        if output_size is None or key_size + output_size > self._budget:
            return output
        if not _finite(output):
            return output
        retained_size = key_size + output_size
        stored = _copy_payload(output)
        if type(output) is np.ndarray and (
            stored.strides != output.strides
            or stored.flags.aligned != output.flags.aligned
        ):
            # Reduction kernels can depend on layout.  Packed native views
            # usually retain their layout with order='K'; holes, negative
            # strides and broadcast strides must remain on the original path.
            return output
        while self._entries and (
            self._retained_bytes + retained_size > self._budget
            or len(self._entries) >= _MAX_ENTRIES
        ):
            _, (_, removed_size) = self._entries.popitem(last=False)
            self._retained_bytes -= removed_size
        self._entries[key] = (stored, retained_size)
        self._retained_bytes += retained_size
        return output

    def _clear_entries(self) -> None:
        self._entries.clear()
        self._retained_bytes = 0
        self._generation += 1

    def close(self) -> None:
        """Release local entries; leave the underlying propagator untouched.

        Further calls delegate without caching.  Closing an already closed
        bank is safe.  This does not invoke the underlying ``clear_cache``.
        """
        self._clear_entries()
        self._closed = True

    def __getstate__(self) -> dict[str, Any]:
        # The original object retains its own serialization policy.  Numeric
        # bank contents and parent diagnostic counters never enter a worker.
        return {
            "source": self._source,
            "budget": self._budget,
            "contract": self._contract,
            "closed": self._closed,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        _PropagatorLookupCache.__init__(
            self, state["source"], state["budget"], state["contract"]
        )
        self._closed = state["closed"]


def wrap_propagator_cache(
    cache: Any,
    propagator_cache_bytes: int | None = None,
    propagator_cache_contract: str | None = None,
) -> Any:
    """Return the original cache when disabled, otherwise a local proxy.

    An active matching bank is reused.  Reconfiguring an active bank with a
    different positive budget is rejected rather than stacking proxies with
    inconsistent accounting.  A closed bank can start a new independent scope.
    """
    budget = validate_cache_options(propagator_cache_bytes, propagator_cache_contract)
    if not budget:
        return cache
    if type(cache) is _PropagatorLookupCache:
        if cache._closed:
            cache = cache._source
        elif cache._budget == budget and cache._contract == propagator_cache_contract:
            return cache
        else:
            raise ValueError(
                "cannot change propagator cache options in an active scope"
            )
    return _PropagatorLookupCache(cache, budget, "fixed_model")
