"""Ordered-word reuse and explicit selected-statistic preparation boundaries."""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
from itertools import product
import math
import pickle
import struct

import numpy as np
import pytest

from sft_wick import quadratic as module
from sft_wick.quadratic import GaussianQuadraticFamily, GaussianQuadraticStatistics


def _bits(value: float) -> bytes:
    return struct.pack("!d", value)


def _family(count=4, latent=3):
    random = np.random.default_rng(271828)
    data = random.normal(size=(count, latent, latent))
    matrices = data @ np.swapaxes(data, -1, -2) / 8
    endpoints = random.normal(size=(count, latent)) / 3
    family = GaussianQuadraticFamily.from_factor(
        np.eye(latent), matrices, endpoints, constant=np.arange(count) / 8
    )
    if latent > 1 and count > 1:
        assert not np.array_equal(matrices[0] @ matrices[1], matrices[1] @ matrices[0])
    return family


def _partitions(items):
    if not items:
        yield []
        return
    first, *rest = items
    for partition in _partitions(rest):
        yield [(first,), *partition]
        for index, block in enumerate(partition):
            yield [*partition[:index], (first, *block), *partition[index + 1 :]]


def _quadrature_cumulant(family, slots):
    # GH5 exactly integrates degree-eight polynomials in exact arithmetic;
    # floating evaluation supplies an independent low-dimensional oracle.
    nodes, weights = np.polynomial.hermite.hermgauss(5)
    latent = family._matrices.shape[-1]
    points = np.array(list(product(nodes * np.sqrt(2), repeat=latent)))
    probabilities = np.array(
        [math.prod(row) for row in product(weights / np.sqrt(np.pi), repeat=latent)]
    )
    values = np.einsum("ni,aij,nj->na", points, family._matrices, points)
    values += points @ family._endpoints.T + family._constants
    result = 0.0
    for partition in _partitions(tuple(range(len(slots)))):
        moments = [
            float(probabilities @ np.prod(values[:, [slots[i] for i in block]], axis=1))
            for block in partition
        ]
        result += (
            (-1) ** (len(partition) - 1)
            * math.factorial(len(partition) - 1)
            * math.prod(moments)
        )
    return result


@pytest.mark.parametrize(
    "slots", [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 1), (0, 0, 1, 2), (0, 1, 2, 3)]
)
@pytest.mark.parametrize("divisor", [1, 2, 4, 6, 24])
def test_ordered_native_reuse_matches_independent_quadrature(slots, divisor):
    family = _family()
    expected = _quadrature_cumulant(family, slots) / divisor
    assert family._mixed_statistic(slots, divisor) == pytest.approx(
        expected, rel=3e-12, abs=3e-12
    )


class _CountingArray(np.ndarray):
    calls = 0

    def __matmul__(self, other):
        type(self).calls += 1
        return super().__matmul__(other)


@pytest.mark.parametrize(
    "slots, words",
    [
        ((0, 0, 0, 0), 1),
        ((0, 0, 0, 1), 4),
        ((0, 0, 1, 1), 6),
        ((0, 0, 1, 2), 12),
        ((0, 1, 2, 3), 24),
    ],
)
def test_only_identical_ordered_words_skip_dense_contractions(slots, words):
    family = _family()
    object.__setattr__(family, "_matrices", family._matrices.view(_CountingArray))
    _CountingArray.calls = 0
    family.mixed_cumulant(slots)
    # Three left-associated ring products and two endpoint matvecs per word.
    assert _CountingArray.calls == 5 * words


def test_reuse_keeps_all_permutation_scalar_and_bound_entries(monkeypatch):
    from sft_wick import _mixed_contraction as contraction

    family = _family()
    observed_terms, observed_classes = [], []
    old_sum, old_classes = module._exact_signed_sum, contraction.word_classes

    def terms(values):
        values = tuple(values)
        observed_terms.append(values)
        return old_sum(values)

    def classes(slots, **kwargs):
        result = old_classes(slots, **kwargs)
        observed_classes.append(result)
        return result

    monkeypatch.setattr(module, "_exact_signed_sum", terms)
    monkeypatch.setattr(contraction, "word_classes", classes)
    family.mixed_cumulant([0, 0, 1, 1])
    assert [len(values) for values in observed_terms] == [24 * 3, 24 * 3]
    assert all(type(value) is float for values in observed_terms for value in values)
    # The rounding certificate weights its ring and chain classes by every
    # one of the 24 slot permutations, repeated words included.
    assert [sum(result.values()) for result in observed_classes] == [24, 24]


def test_selected_bank_deduplicates_exact_keys_but_preserves_order(monkeypatch):
    family = _family(2)
    cumulants, coefficients = [], []
    old_mixed, old_coefficient = family.mixed_cumulant, family.joint_coefficient

    def mixed(self, indices):
        cumulants.append(tuple(indices))
        return old_mixed(indices)

    def coefficient(self, powers):
        coefficients.append(tuple(powers))
        return old_coefficient(powers)

    monkeypatch.setattr(GaussianQuadraticFamily, "mixed_cumulant", mixed)
    monkeypatch.setattr(GaussianQuadraticFamily, "joint_coefficient", coefficient)
    selected = family.prepare_statistics(
        indices=[(1, 0), (0, 1), (1, 0)],
        coefficients=[(2, 0), (2, 0), (0, 0)],
        max_entries=4,
    )
    assert cumulants == [(1, 0), (0, 1)]
    assert coefficients == [(2, 0), (0, 0)]
    assert selected.entry_count == 4
    assert selected.observable_count == 2
    assert selected.indices == ((0, 1), (1, 0))
    assert selected.coefficients == ((0, 0), (2, 0))
    assert _bits(selected.mixed_cumulant([1, 0])) == _bits(old_mixed([1, 0]))
    assert _bits(selected.joint_coefficient([2, 0])) == _bits(old_coefficient([2, 0]))
    assert selected.joint_coefficient([0, 0]) == 0.0


@pytest.mark.parametrize("latent", [0, 1, 3])
def test_bank_selected_queries_are_bitwise_original_and_do_no_warm_work(
    latent, monkeypatch
):
    family = _family(4, latent)
    indices = [(0,), (1, 0), (0, 0, 1), (0, 0, 1, 1), (0, 1, 2, 3)]
    powers = [(0, 0, 0, 0), (2, 0, 0, 0), (0, 1, 2, 1)]
    expected_mixed = [_bits(family.mixed_cumulant(key)) for key in indices]
    expected_coefficients = [_bits(family.joint_coefficient(key)) for key in powers]
    selected = family.prepare_statistics(indices=indices, coefficients=powers)

    def forbidden(*args, **kwargs):
        raise AssertionError("a prepared lookup ran a dense contraction")

    monkeypatch.setattr(GaussianQuadraticFamily, "_mixed_statistic", forbidden)
    for key, expected in zip(indices, expected_mixed):
        assert _bits(selected.mixed_cumulant(key)) == expected
    for key, expected in zip(powers, expected_coefficients):
        assert _bits(selected.joint_coefficient(key)) == expected
    assert all(
        not isinstance(value, GaussianQuadraticFamily)
        for value in (selected._keys, selected._outcomes, selected._values)
    )
    assert selected._values.nbytes == 8 * selected.entry_count


def test_cumulant_overflow_and_finite_coefficient_are_prepared_independently():
    family = GaussianQuadraticFamily.from_factor([[1.0]], [[[1e154]]])
    with pytest.raises(OverflowError) as original:
        family.mixed_cumulant([0, 0])
    finite = family.joint_coefficient([2])
    selected = family.prepare_statistics(indices=[(0, 0)], coefficients=[(2,)])
    with pytest.raises(OverflowError) as prepared:
        selected.mixed_cumulant([0, 0])
    assert str(prepared.value) == str(original.value)
    assert _bits(selected.joint_coefficient([2])) == _bits(finite)
    assert selected._values.nbytes == 8
    assert selected._outcomes[0] == (OverflowError, str(original.value))


def test_exact_work_bound_failure_is_immutable_record_and_rethrown(monkeypatch):
    from sft_wick import _mixed_contraction as contraction

    matrices = np.zeros((2, 64, 64))
    matrices[0, :3, :3] = np.diag([1e150, -1e150, 1.0])
    matrices[1, :3, :3] = np.diag([2e150, -2e150, 3.0])
    family = GaussianQuadraticFamily.from_factor(np.eye(64), matrices)
    # tr(A*A*B) = 3 exactly; the exact contraction returns it below the
    # work bound and the bound is lowered here to exercise the refusal.
    assert family.mixed_cumulant([0, 0, 1]) == 24.0
    monkeypatch.setattr(contraction, "EXACT_WORK_LIMIT", 1000)
    with pytest.raises(ArithmeticError) as original:
        family.mixed_cumulant([0, 0, 1])
    selected = family.prepare_statistics(indices=[(0, 0, 1)])
    assert selected._values.nbytes == 0
    assert selected._outcomes == ((ArithmeticError, str(original.value)),)
    for restored in (
        selected,
        copy.copy(selected),
        copy.deepcopy(selected),
        pickle.loads(pickle.dumps(selected)),
    ):
        with pytest.raises(ArithmeticError) as prepared:
            restored.mixed_cumulant([0, 0, 1])
        assert type(prepared.value) is type(original.value)
        assert str(prepared.value) == str(original.value)


@pytest.mark.parametrize(
    "transform",
    [copy.copy, copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))],
)
def test_bank_copy_pickle_and_all_arrays_remain_immutable(transform):
    family = _family(2)
    selected = family.prepare_statistics(indices=[(0,), (1, 0)], coefficients=[(2, 0)])
    restored = transform(selected)
    assert restored._keys == selected._keys
    assert restored._outcomes == selected._outcomes
    assert np.array_equal(restored._values, selected._values)
    with pytest.raises(FrozenInstanceError):
        restored._observable_count = 999
    with pytest.raises(TypeError):
        restored._keys[0] = (0, (1,))
    with pytest.raises(ValueError):
        restored._values[0] = 999
    with pytest.raises(ValueError):
        restored._values.setflags(write=True)
    base = restored._values
    while isinstance(base, np.ndarray):
        assert not base.flags.writeable
        with pytest.raises(ValueError):
            base.setflags(write=True)
        base = base.base
    assert isinstance(base, bytes)
    assert _bits(restored.mixed_cumulant([1, 0])) == _bits(
        family.mixed_cumulant([1, 0])
    )


def test_prepared_payload_is_snapshot_and_does_not_alias_mutable_inputs():
    values = np.array([1.25])
    selected = GaussianQuadraticStatistics._from_prepared(1, ((0, (0,)),), (0,), values)
    values[0] = -2
    assert selected.mixed_cumulant([0]) == 1.25


@pytest.mark.parametrize(
    "arguments",
    [
        {"indices": [()]},
        {"indices": [(0,) * 5]},
        {"indices": [(4,)]},
        {"indices": [(-1,)]},
        {"indices": [(True,)]},
        {"indices": [(1.0,)]},
        {"coefficients": [(0,)]},
        {"coefficients": [(5, 0, 0, 0)]},
        {"coefficients": [(False, 0, 0, 0)]},
        {"coefficients": [(-1, 0, 0, 0)]},
        {"max_entries": -1},
        {"max_entries": 1025},
        {"max_entries": True},
        {"max_entries": 1.0},
        {"indices": [(0,), (1,)], "max_entries": 1},
        {"indices": iter([(0,)])},
        {"indices": [iter([0])]},
    ],
)
def test_invalid_preparation_and_budgets_fail_before_any_dense_query(
    arguments, monkeypatch
):
    family = _family()

    def forbidden(*args, **kwargs):
        raise AssertionError("preparation computed before validating all selected keys")

    monkeypatch.setattr(GaussianQuadraticFamily, "mixed_cumulant", forbidden)
    monkeypatch.setattr(GaussianQuadraticFamily, "joint_coefficient", forbidden)
    with pytest.raises(ValueError):
        family.prepare_statistics(**arguments)


def test_key_storage_budget_is_checked_before_consuming_oversized_query():
    class Oversized:
        def __len__(self):
            return 65_537

        def __iter__(self):
            raise AssertionError("an oversized query was consumed")

    with pytest.raises(ValueError, match="key storage budget"):
        _family().prepare_statistics(indices=[Oversized()])


def test_duplicate_count_is_distinct_from_unique_entry_budget():
    selected = _family().prepare_statistics(indices=[(0,)] * 2048, max_entries=1)
    assert selected.entry_count == 1
    empty = _family().prepare_statistics(max_entries=0)
    assert empty.entry_count == 0
    assert empty._values.nbytes == 0
    with pytest.raises(KeyError, match="not prepared"):
        empty.mixed_cumulant([0])


def test_exact_max_entries_boundary_is_supported_and_rejected_upfront(monkeypatch):
    family = _family(6, 0)
    keys = list(product(range(6), repeat=4))
    calls = []

    def selected(self, indices):
        calls.append(tuple(indices))
        return 1.25

    monkeypatch.setattr(GaussianQuadraticFamily, "mixed_cumulant", selected)
    prepared = family.prepare_statistics(indices=keys[:1024])
    assert prepared.entry_count == 1024
    assert len(calls) == 1024
    assert prepared.mixed_cumulant(keys[1023]) == 1.25
    calls.clear()
    with pytest.raises(ValueError, match="max_entries"):
        family.prepare_statistics(indices=keys[:1025])
    assert calls == []


def test_cumulative_key_budget_rejects_before_any_numeric_query(monkeypatch):
    count = 256
    family = _family(count, 0)
    powers = [
        tuple(int(index == observable) for index in range(count))
        for observable in range(count)
    ]
    powers.append((2,) + (0,) * (count - 1))

    def forbidden(*args, **kwargs):
        raise AssertionError("key budget checked after a dense query")

    monkeypatch.setattr(GaussianQuadraticFamily, "joint_coefficient", forbidden)
    with pytest.raises(ValueError, match="key storage budget"):
        family.prepare_statistics(coefficients=powers)


def test_mutable_query_inputs_are_snapshotted_to_immutable_keys():
    indices, powers = [[0, 1]], [[1, 0]]
    prepared = _family(2).prepare_statistics(indices=indices, coefficients=powers)
    expected = prepared.mixed_cumulant([0, 1])
    indices[0][:] = [1, 0]
    powers[0][:] = [0, 1]
    assert prepared.indices == ((0, 1),)
    assert prepared.coefficients == ((1, 0),)
    assert _bits(prepared.mixed_cumulant([0, 1])) == _bits(expected)


@pytest.mark.parametrize("inner", [False, True])
@pytest.mark.parametrize("declared,actual", [(1, 0), (1, 2), (2, 1), (2, 3)])
def test_each_sequence_layer_has_bounded_consumption_and_exact_length(
    inner, declared, actual, monkeypatch
):
    consumed = []

    class Misdeclared:
        def __len__(self):
            return declared

        def __iter__(self):
            for index in range(actual):
                consumed.append(index)
                yield 0 if inner else (0,)

    family = _family()

    def forbidden(*args, **kwargs):
        raise AssertionError("misdeclared input triggered a numeric query")

    monkeypatch.setattr(GaussianQuadraticFamily, "mixed_cumulant", forbidden)
    query = [Misdeclared()] if inner else Misdeclared()
    with pytest.raises(ValueError, match="declared sequence length"):
        family.prepare_statistics(indices=query)
    assert len(consumed) == min(actual, declared + 1)


@pytest.mark.parametrize("inner", [False, True])
def test_infinite_sequence_iterator_is_rejected_after_one_extra_item(inner):
    consumed = []

    class Infinite:
        def __len__(self):
            return 1

        def __iter__(self):
            while True:
                consumed.append(len(consumed))
                assert len(consumed) <= 2
                yield 0 if inner else (0,)

    with pytest.raises(ValueError, match="declared sequence length"):
        _family().prepare_statistics(indices=[Infinite()] if inner else Infinite())
    assert len(consumed) == 2


def test_total_submitted_query_budget_is_checked_before_outer_consumption():
    class Oversized:
        def __init__(self, length):
            self.length = length

        def __len__(self):
            return self.length

        def __iter__(self):
            raise AssertionError("outer submission budget checked after consumption")

    family = _family()
    with pytest.raises(ValueError, match="submitted query budget"):
        family.prepare_statistics(indices=Oversized(65_537))
    with pytest.raises(ValueError, match="submitted query budget"):
        family.prepare_statistics(
            indices=Oversized(32_768), coefficients=Oversized(32_769)
        )


@pytest.mark.parametrize("kind,length", [("indices", 5), ("coefficients", 1)])
def test_invalid_query_shape_rejected_before_inner_consumption(kind, length):
    class Invalid:
        def __len__(self):
            return length

        def __iter__(self):
            raise AssertionError("invalid key shape consumed before rejection")

    with pytest.raises(ValueError):
        _family().prepare_statistics(**{kind: [Invalid()]})


def test_cross_kind_keys_and_unprepared_queries_cannot_collide():
    family = _family(2)
    selected = family.prepare_statistics(indices=[(0, 1)], coefficients=[(1, 0)])
    assert selected.entry_count == 2
    with pytest.raises(KeyError, match="not prepared"):
        selected.mixed_cumulant([1, 0])
    with pytest.raises(KeyError, match="not prepared"):
        selected.joint_coefficient([0, 1])
    with pytest.raises(ValueError, match="integer"):
        selected.mixed_cumulant([True])
    with pytest.raises(ValueError, match="orders one through four"):
        selected.mixed_cumulant([])
    with pytest.raises(ValueError, match="out of range"):
        selected.mixed_cumulant([2])
    with pytest.raises(ValueError, match="total degree"):
        selected.joint_coefficient([4, 1])
    assert not hasattr(selected, "cgf")
    assert not hasattr(selected, "project")


@pytest.mark.parametrize(
    "error",
    [
        ValueError("invalid"),
        RuntimeError("unexpected"),
        FloatingPointError("unexpected numeric"),
    ],
)
def test_unexpected_exceptions_are_not_cached(error, monkeypatch):
    def fail(self, indices):
        raise error

    monkeypatch.setattr(GaussianQuadraticFamily, "mixed_cumulant", fail)
    with pytest.raises(type(error)) as actual:
        _family().prepare_statistics(indices=[(0,)])
    assert actual.value is error


def test_no_directional_polynomial_reconstruction_is_promised():
    family = GaussianQuadraticFamily.from_factor(
        np.eye(2), [np.diag([1e16, 1.0]), np.diag([-1e16, 0.0])]
    )
    selected = family.prepare_statistics(indices=[(0,), (1,)])
    assert family.project([1.0, 1.0]).mean == 1.0
    assert selected.mixed_cumulant([0]) + selected.mixed_cumulant([1]) == 0.0
    assert not hasattr(selected, "directional_cumulant")


def test_direct_selected_bank_construction_has_explicit_preparation_instruction():
    with pytest.raises(TypeError, match="prepare_statistics"):
        GaussianQuadraticStatistics()


def test_family_subclasses_are_rejected_before_overridden_queries_run():
    class Stateful(GaussianQuadraticFamily):
        def mixed_cumulant(self, indices):
            raise AssertionError("a subclass query ran")

    family = Stateful.from_factor([[1.0]], [[[1.0]]])
    with pytest.raises(TypeError, match="native GaussianQuadraticFamily"):
        family.prepare_statistics(indices=[(0,)])


@pytest.mark.parametrize(
    "error",
    [OverflowError(3), ArithmeticError("a", "b"), ZeroDivisionError("unexpected")],
)
def test_numeric_exception_schema_is_not_silently_changed(error, monkeypatch):
    def fail(self, indices):
        raise error

    monkeypatch.setattr(GaussianQuadraticFamily, "mixed_cumulant", fail)
    with pytest.raises(type(error)) as actual:
        _family().prepare_statistics(indices=[(0,)])
    assert actual.value is error
    assert actual.value.args == error.args


@pytest.mark.parametrize(
    "matrices,slots,powers",
    [
        ([np.diag([1e-120, -1e-120, 1e-120])], (0,) * 4, (4,)),
        ([np.diag([1e100, -1e100, 1.0])], (0,) * 3, (3,)),
        ([np.zeros((3, 3))], (0,) * 4, (4,)),
    ],
)
def test_prepared_underflow_zero_and_cancellation_preserve_original_bits(
    matrices, slots, powers
):
    family = GaussianQuadraticFamily.from_factor(np.eye(3), matrices)
    expected_mixed = family.mixed_cumulant(slots)
    expected_coefficient = family.joint_coefficient(powers)
    selected = family.prepare_statistics(indices=[slots], coefficients=[powers])
    assert _bits(selected.mixed_cumulant(slots)) == _bits(expected_mixed)
    assert _bits(selected.joint_coefficient(powers)) == _bits(expected_coefficient)
