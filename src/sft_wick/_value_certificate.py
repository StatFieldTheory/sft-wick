"""Evaluate emitted terms and an independent pairing sum on shared fixtures."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations_with_replacement, product
from math import factorial, isfinite, prod

import numpy as np

from ._value_reference import Slot, ValueLayoutError, contract, numerical_sum, reference_value
from .expressions import ImaginaryUnit, KroneckerDelta, Product, Rational, Sum, Symbol


MAX_OPERATORS = 14  # (14-1)!! = 135,135 candidate pairings
VALUE_RTOL = 1e-12
MAX_TENSOR_ENTRIES = 3**MAX_OPERATORS


@dataclass(frozen=True)
class ValueComparison:
    """Complex values as (real, imaginary) pairs, in a named probe."""

    probe: str
    reference: tuple[float, float]
    emitted: tuple[float, float]
    relative_error: float

    @property
    def passed(self) -> bool:
        return isfinite(self.relative_error) and self.relative_error <= VALUE_RTOL


def sequence_prefactors(n_vertices, order):
    """MSR series coefficients from selections of action slots, not engine terms."""
    for selection in combinations_with_replacement(range(n_vertices), order):
        repeated = Counter(selection).values()
        yield (-1) ** order / prod(factorial(count) for count in repeated)


def _literal(label):
    try:
        return int(label) - 1
    except (ValueError, TypeError):
        return None


def _monomials(expr):
    """The products of coupling factors, including factors wrapped around a sum."""
    if isinstance(expr, Sum):
        for child in expr.terms:
            yield from _monomials(child)
    elif isinstance(expr, Product):
        for selection in product(*map(_monomials, expr.factors)):
            yield tuple(factor for term in selection for factor in term)
    else:
        yield (expr,)


class ValueFixture:
    """One sequence, realized without using instantiated engine operators."""

    def __init__(self, observable, vertices, flags):
        diag_R, diag_C, iso_R, iso_C = flags
        self.diagonal = {"R": diag_R or iso_R, "C": diag_C or iso_C}
        self.isotropic = {"R": iso_R, "C": iso_C}
        literals = {_literal(op.component_index) for op in observable}
        bases = sorted({0, 1, 2} | {i for i in literals if i is not None and i >= 0})
        self.basis = tuple(bases)
        self.width = len(self.basis)
        self.identity = np.eye(self.width)
        self.rng = np.random.default_rng(5719)
        self.couplings = {}
        self.external_points = {}
        self.external_indices = {}
        self.roles = []
        self.vertex_roles = {}
        self.vertex_points = []
        self.slots = []
        self.reference_factors = []
        self.n_owners = len(observable) + len(vertices)
        self._make_slots(observable, vertices)
        self.matrices = {}

    def _dimension(self, declared):
        return sum(component < declared for component in self.basis)

    def _make_slots(self, observable, vertices):
        index_widths = {}
        for op in observable:
            index = op.component_index
            if index is not None:
                width = self._dimension(op.field.n_components)
                index_widths[index] = min(width, index_widths.get(index, width))
        free = 0
        for index, width in index_widths.items():
            literal = _literal(index)
            if literal is None:
                self.external_indices[index] = free % width
                free += 1
            else:
                self.external_indices[index] = self.basis.index(literal)
        for owner, op in enumerate(observable):
            label = op.spatial_arg
            if label not in self.external_points:
                self.external_points[label] = len(self.roles)
                self.roles.append(len(self.roles))
            point = self.external_points[label]
            fixed = self.external_indices.get(op.component_index, 0)
            self.slots.append(Slot(point, owner, op.is_response, None, fixed, 1,
                                   op.field.n_components))
        for owner, vertex in enumerate(vertices, len(observable)):
            role = self.vertex_roles.setdefault(vertex.coupling, len(self.roles))
            point_count = 1 if vertex.local else len(vertex.fields)
            points = tuple(range(len(self.roles), len(self.roles) + point_count))
            self.roles.extend([role] * point_count)
            self.vertex_points.append(points)
            indices, shape = [], []
            for leg, field in enumerate(vertex.fields):
                point = points[0 if vertex.local else leg]
                axis = None if field.is_scalar else len(self.slots)
                width = self._dimension(field.n_components)
                self.slots.append(Slot(point, owner, field.is_response, axis,
                                       0 if axis is None else None, width,
                                       field.n_components, vertex.already_R_contracted))
                if axis is not None:
                    indices.append(axis)
                    shape.append(width)
            if prod(shape) > MAX_TENSOR_ENTRIES:
                raise ValueLayoutError("coupling tensor exceeds the numeric probe budget")
            if vertex.coupling not in self.couplings:
                tensor = self.rng.uniform(-1, 1, shape) + 1j * self.rng.uniform(-1, 1, shape)
                self.couplings[vertex.coupling] = np.asarray(tensor)
            tensor = self.couplings[vertex.coupling]
            if tensor.shape != tuple(shape):
                raise ValueLayoutError("one coupling name has multiple tensor shapes")
            self.reference_factors.append((tensor, tuple(indices)))

    def matrix(self, probe, kind, a, b, absorbed=False, roles=None):
        if absorbed:
            return self.identity
        same = a == b
        if probe == "roles":
            if roles is None:
                roles = self.roles
            a, b = roles[a], roles[b]
        transpose = kind == "C" and a > b
        if transpose:
            a, b = b, a
        key = probe, kind, a, b, same
        if key not in self.matrices:
            if self.isotropic[kind]:
                matrix = self.identity * self.rng.uniform(-1, 1)
            else:
                matrix = self.rng.uniform(-1, 1, (self.width, self.width))
                if kind == "R":
                    matrix = matrix + 1j * self.rng.uniform(-1, 1, matrix.shape)
                if self.diagonal[kind]:
                    matrix = np.diag(np.diag(matrix))
                if kind == "C" and a == b:
                    matrix = (matrix + matrix.T) * 0.5
            self.matrices[key] = matrix
        matrix = self.matrices[key]
        return matrix.T if transpose else matrix

    def engine_points(self, instances):
        """Bridge actual engine labels to the independently constructed points."""
        points = self.external_points.copy()
        for instance, labels in zip(instances, self.vertex_points, strict=True):
            points.update(zip(instance.spatial_variables, labels, strict=True))
        return points

    def reference(self, probe, prefactor, target):
        return reference_value(
            self.slots, self.reference_factors, self.n_owners, target == "connected",
            prefactor, lambda kind, a, b, absorbed: self.matrix(probe, kind, a, b, absorbed),
            self.diagonal,
        )

    def emitted(self, terms, instances, probe, grouped):
        if not terms:
            return 0j
        points = self.engine_points(instances)
        return numerical_sum(self._term(term, points, probe, grouped) for term in terms)

    def _term(self, term, points, probe, grouped):
        axes = {index: (i, self._dimension(dim))
                for i, (index, dim) in enumerate(term.summation_indices)}

        def factor(array, indices):
            subscripts, free = [], []
            for index in indices:
                if index in axes:
                    axis, width = axes[index]
                    subscripts.append(slice(width))
                    free.append(axis)
                else:
                    subscripts.append(self.external_indices[index] if index is not None else 0)
            return np.asarray(array[tuple(subscripts)]), tuple(free)

        def values():
            for monomial in _monomials(term.coupling_sum):
                # Grouped terms may relabel internal points across different
                # vertex types. Each monomial carries the resulting roles.
                roles = self.roles.copy() if grouped else None
                factors = []
                used = set()
                for leaf in monomial:
                    if isinstance(leaf, Symbol):
                        entry = factor(self.couplings[leaf.name], leaf.indices)
                        if roles is not None:
                            for point in leaf.spatial_args:
                                roles[points[point]] = self.vertex_roles[leaf.name]
                    elif isinstance(leaf, KroneckerDelta):
                        entry = factor(self.identity, (leaf.index1, leaf.index2))
                    elif isinstance(leaf, Rational):
                        entry = (np.asarray(leaf.numerator / leaf.denominator), ())
                    elif isinstance(leaf, ImaginaryUnit):
                        entry = (np.asarray(1j), ())
                    else:
                        raise TypeError(f"unreadable coupling node {type(leaf).__name__}")
                    factors.append(entry)
                    used.update(entry[1])
                for edge in term.propagators:
                    labels = edge.spatial_left, edge.spatial_right
                    # Absorbed factors are skipped by numerical evaluation.
                    # Reapplying the identity here would hide missing pins.
                    if edge.kind == "R" and labels in term.r_absorbed_pairs:
                        continue
                    array = self.matrix(probe, edge.kind, points[labels[0]], points[labels[1]],
                                        roles=roles)
                    entry = factor(array, (edge.index_left, edge.index_right))
                    factors.append(entry)
                    used.update(entry[1])
                # An index that was retained by the engine but is absent in
                # all factors still contributes the sum of 1 over that axis.
                multiplicity = prod(width for axis, width in axes.values() if axis not in used)
                yield multiplicity * contract(factors)

        prefactor = term.rational_prefactor
        engine_phase = term.response_phase_factor()
        return (prefactor.numerator / prefactor.denominator * engine_phase
                * numerical_sum(values()))


def compare_values(observable, vertices, target, prefactor, terms, instances, flags, engine):
    fixture = ValueFixture(observable, vertices, flags)
    probes = ("roles", "points") if engine == "compute_moment_numerical" else ("roles",)
    for probe in probes:
        reference = fixture.reference(probe, prefactor, target)
        emitted = fixture.emitted(terms, instances, probe, engine == "compute_moment")
        reference_size, emitted_size = abs(reference), abs(emitted)
        error = float("inf")
        if isfinite(reference_size) and isfinite(emitted_size):
            scale = max(reference_size, emitted_size)
            error = abs(reference - emitted) / scale if scale else 0.0
        yield ValueComparison(probe, (reference.real, reference.imag),
                              (emitted.real, emitted.imag), error)
