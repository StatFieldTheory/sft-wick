"""The ``"factorized"`` backend: contract the original coupling factors.

Preparation keeps owned copies of the static coupling tensors of a native
monomial or sum of monomials instead of the dense coefficient over all
propagator component indices.  Each call contracts every monomial with the
propagator batches and adds the monomial results, under the accuracy and
behaviour contract of :mod:`sft_wick._dense_contraction`.  When a call
declines, or when the public ``coupling_array`` is read, the ordered
preparation (``DiagramTerm.evaluate_coupling``) builds the dense coefficient
from the owned copies and the integrand continues on the ordered fold.

A sum of monomials is prepared only when no two monomials have opposite
signs at any component assignment.  Monomials that cancel would leave the
contracted result with an error proportional to the sum of the monomial
magnitudes, while the ordered fold sums the monomials into one coefficient
before it multiplies the propagators.  The sign check runs once, on sign
arrays of the factors, and also counts the nonzero coefficients for the
cost model.  When that per-assignment check would take too much work, the
sum is prepared only if the factor signs alone prove agreement (every
factor holds one sign apart from zeros, and every monomial has the same
resulting sign), and the nonzero count is replaced by a lower bound.

The contraction budget is a conservative ndarray payload, never process RSS.
Admission accepts the package's own classes and exact float64 or complex128
coupling arrays; replacing package methods at run time is unsupported.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from threading import Lock

import numpy as np

from . import _dense_contraction as _dense
from . import perturbation as _pt
from ._coupling_plan import _CouplingPlan
from ._dense_contraction import (
    QUARTER_TURNS,
    errors_allow_backend,
    factorized_overhead,
    finish,
    finite_range,
    fixed_labels_ok,
    kernel_layout,
    live_assignments,
    minimum_backend_cost,
    ordered_cost,
    plan_groups,
    _native_dtype,
    _ordinary_label,
)
from .expressions import _NATIVE_NUMERIC_EXPRESSION_CLASSES

# Definition-time class bindings, as the ordered numerical evaluators use.
(_, _, Product, Sum, Symbol, Rational, ImaginaryUnit, Propagator) = (
    _NATIVE_NUMERIC_EXPRESSION_CLASSES
)

#: Payload reserved beside a materialized coefficient.
_SCAN_RESERVE = 64 * 4096
#: Element operations the build-time per-assignment sign check may spend
#: (monomials times assignments times factors).  Above it a sum is prepared
#: only when the signs of its factors prove agreement.
_MAX_SIGN_WORK = 2**27
#: Assignments evaluated together by the sign check.
_SIGN_CHUNK = 2**20
#: Largest coefficient whose 0/1 zero pattern is kept for the cost model.
_MASK_ELEMENTS = 2**16
#: Lowering limits: expression depth and factors per monomial.
_MAX_DEPTH = 32
_MAX_FACTORS = 24

_CLASS = None
_CLASS_LOCK = Lock()


@dataclass(frozen=True)
class _Monomial:
    #: ``(slot, imaginary, slices, labels)`` per tensor factor.
    factors: tuple
    #: Real scalar: prefactor, rational factors and phases of this monomial.
    phase: float
    #: ``(magnitude, magnitude)`` of the prefactor and of each rational
    #: factor whose magnitude is not 1.  The ordered preparation multiplies
    #: them one at a time, so the range rules count each as a factor of its
    #: own rather than their folded product ``phase``.
    scales: tuple


@dataclass(frozen=True)
class _Snapshot:
    #: Coupling names and owned read-only copies of their arrays.
    names: tuple
    arrays: tuple
    #: :func:`~sft_wick._dense_contraction.finite_range` of each array's
    #: used component (its real or imaginary part).
    ranges: tuple
    monomials: tuple
    prop_idx: tuple
    fixed: tuple
    #: Coefficient dtype of the ordered preparation (``float`` or ``complex``).
    dtype: type
    e_psi: int
    nbytes: int
    #: Nonzero entries of the summed coefficient, or a lower bound (the
    #: largest count of one monomial) when the per-assignment check was not
    #: run.
    nonzero: int
    #: Its 0/1 float64 zero pattern over the propagator labels, or ``None``.
    mask: object


class _CoefficientOwner:
    """One latent ndarray binding shared by configured shallow views."""

    def __init__(self, array, snapshot=None, present=True):
        self.array = array
        self.snapshot = snapshot
        self.present = present
        self.lock = Lock()

    def __getstate__(self):
        with self.lock:
            return self.array, self.snapshot, self.present

    def __setstate__(self, state):
        self.array, self.snapshot, self.present = state
        self.lock = Lock()
        if self.snapshot is not None:
            for array in self.snapshot.arrays:
                array.flags.writeable = False
            if self.snapshot.mask is not None:
                self.snapshot.mask.flags.writeable = False


def share_factorized_owner(view, source):
    """Bind a configured shallow view to the same latent coefficient ndarray.

    Reading either view materializes once and disables fusion for both. Field
    replacement/deletion detaches only that instance, as with ordinary shallow
    copies of an ndarray field. This helper performs no public array read.
    """
    if type(view) is not _CLASS or type(source) is not _CLASS:
        return
    source_state = object.__getattribute__(source, "__dict__")
    view_state = object.__getattribute__(view, "__dict__")
    owner = source_state.get("_factorized_owner")
    if type(owner) is not _CoefficientOwner:
        return
    with owner.lock:
        view_state["_factorized_owner"] = owner
        view_state["_factorized_plan"] = owner.snapshot
        view_state["_factorized_lock"] = owner.lock
        if owner.present:
            view_state["coupling_array"] = owner.array
        else:
            view_state.pop("coupling_array", None)


def factorized_state(integrand):
    """Read authoritative private state for untimed acceptance instrumentation."""
    state = object.__getattribute__(integrand, "__dict__")
    owner = state.get("_factorized_owner")
    if type(owner) is not _CoefficientOwner:
        return state.get("coupling_array"), state.get("_factorized_plan")
    with owner.lock:
        return owner.array if owner.present else None, owner.snapshot


#: A coupling name no expression can use; it only carries the complex dtype
#: of coupling values that are not part of the snapshot.
#: ``DiagramTerm.evaluate_coupling`` makes its result complex when any value
#: of the coupling dict is complex, used or not; the materialization test
#: with an unused complex value pins that rule.
_DTYPE_WITNESS = " complex"


def _materialize(snapshot, term, max_bytes):
    """The coefficient the ordered preparation builds, from the snapshot.

    Runs ``DiagramTerm.evaluate_coupling`` on the owned copies, so the array
    equals the ordered one in dtype and in every bit.
    """
    shape = tuple(dim for _, dim in snapshot.prop_idx)
    planned = (
        snapshot.nbytes
        + 2 * math.prod(shape) * np.dtype(snapshot.dtype).itemsize
        + _SCAN_RESERVE
    )
    _require_materialization_budget(planned, max_bytes)
    values = dict(zip(snapshot.names, snapshot.arrays))
    if snapshot.dtype is complex and not any(
        _native_dtype(array, np.complex128) for array in snapshot.arrays
    ):
        values[_DTYPE_WITNESS] = np.zeros((), dtype=np.complex128)
    return term.evaluate_coupling(values, dict(snapshot.fixed))


def _require_materialization_budget(planned, max_bytes):
    if type(max_bytes) is not int or max_bytes <= 0 or planned > max_bytes:
        raise ValueError(
            "factorized coefficient materialization requires a conservative "
            f"ndarray payload of {planned} bytes, exceeding "
            f"contraction_max_bytes={max_bytes}; increase this budget or rebuild "
            "this integrand with contraction_backend='ordered' explicitly to permit the original "
            "coefficient materialization"
        )


def _native_term(term) -> bool:
    """A package ``DiagramTerm`` with ordinary labels and no foreign plan."""
    if type(term) is not _pt.DiagramTerm:
        return False
    state = object.__getattribute__(term, "__dict__")
    plan = state.get("_coupling_execution_plan")
    if plan is not None and type(plan) is not _CouplingPlan:
        return False
    summation = state.get("summation_indices")
    propagators = state.get("propagators")
    return (
        type(summation) is tuple
        and all(
            type(item) is tuple
            and len(item) == 2
            and _ordinary_label(item[0])
            and type(item[1]) is int
            and item[1] > 0
            for item in summation
        )
        and type(propagators) is tuple
        and all(
            type(prop) is Propagator
            and all(
                value is None or _ordinary_label(value)
                for value in (prop.index_left, prop.index_right)
            )
            for prop in propagators
        )
    )


def check_materialization_budget(term, static_cv, max_bytes):
    """Reject a known oversized dense fallback without building it.

    Only the factorized builder uses this policy.  The coefficient shape is
    the product of the summation dimensions that appear in a propagator.
    Two complex128 arrays conservatively cover coefficient preparation and
    its final prefactor multiplication.
    """
    if type(term) is not _pt.DiagramTerm:
        return
    summation = term.summation_indices
    propagators = term.propagators
    if type(summation) is not tuple or type(propagators) is not tuple:
        return
    indices = set()
    for prop in propagators:
        if type(prop) is not Propagator:
            return
        for name in (prop.index_left, prop.index_right):
            if name is not None:
                if type(name) is not str:
                    return
                indices.add(name)
    elements = 1
    names = set()
    for item in summation:
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not int
            or item[1] <= 0
            or item[0] in names
        ):
            return
        names.add(item[0])
        if item[0] in indices:
            elements *= item[1]
    _require_materialization_budget(32 * elements + _SCAN_RESERVE, max_bytes)


def _contract(snapshot, fixed, cb, rp, jac, rb, e_psi, max_bytes):
    """Contract every monomial with the propagator batches, or decline."""
    if (
        type(max_bytes) is not int
        or max_bytes <= 0
        or not errors_allow_backend()
        or e_psi != snapshot.e_psi
        or not fixed_labels_ok(fixed)
    ):
        return None
    if (
        type(cb) not in (list, tuple)
        or type(rb) not in (list, tuple)
        or not _dense._float64(rp)
    ):
        return None
    names = {name: (axis, dim) for axis, (name, dim) in enumerate(snapshot.prop_idx, 1)}
    dims = {axis: dim for axis, dim in names.values()}
    samples = rp.size
    assignments = math.prod(dims.values())
    nonzero = snapshot.nonzero
    n_c, n_r = len(cb), len(rb)
    overhead = factorized_overhead(len(snapshot.monomials), samples)
    # The ordered estimate with every nonzero assignment live bounds it from
    # above; the step count of every monomial bounds the backend from below.
    upper = ordered_cost(assignments, nonzero, nonzero, n_c, n_r, samples)
    steps = sum(len(m.factors) + 1 + n_c + n_r for m in snapshot.monomials)
    if not _dense.backend_is_cheaper(overhead + minimum_backend_cost(steps), upper):
        return None
    layout = kernel_layout(cb, rb, rp, jac, names, fixed)
    if layout is None:
        return None
    groups = []
    for monomial in snapshot.monomials:
        operands = []
        for slot, imaginary, slices, labels in monomial.factors:
            array = snapshot.arrays[slot]
            component = array.imag if imaginary else array.real
            operands.append((component[slices], labels))
        groups.append((monomial.phase, tuple(operands)))
    planned = plan_groups(tuple(groups), layout, dims, samples)
    if planned is None:
        return None
    contraction, _ = planned
    if contraction.workspace > max_bytes:
        return None
    cost = contraction.cost + overhead
    if not _dense.backend_is_cheaper(cost, upper):
        return None
    live, count_cost = live_assignments(dims, snapshot.mask, layout.supports)
    if live is None:
        return None
    estimate = ordered_cost(
        assignments, nonzero, min(live, nonzero), layout.n_c, layout.n_r, samples
    )
    if not _dense.backend_is_cheaper(cost + count_cost, estimate):
        return None
    ranges = tuple(
        tuple(snapshot.ranges[slot] for slot, _, _, _ in m.factors) + m.scales
        for m in snapshot.monomials
    )
    return finish(contraction, layout, ranges, assignments, phase_is_factor=False)


def _integrand_class():
    if _CLASS is not None:
        return _CLASS
    with _CLASS_LOCK:
        return _create_integrand_class()


def _create_integrand_class():
    global _CLASS
    if _CLASS is not None:
        return _CLASS
    from .evaluate import DiagramIntegrand

    class _FactorizedIntegrand(DiagramIntegrand):
        @property
        def coupling_array(self):
            state = object.__getattribute__(self, "__dict__")
            while True:
                owner = state.get("_factorized_owner")
                if owner is None:
                    if "coupling_array" not in state:
                        raise AttributeError("coupling_array")
                    return state["coupling_array"]
                with owner.lock:
                    if state.get("_factorized_owner") is not owner:
                        continue
                    if not owner.present:
                        raise AttributeError("coupling_array")
                    if owner.snapshot is not None:
                        owner.array = _materialize(
                            owner.snapshot,
                            state.get("diagram_term"),
                            state.get("_contraction_max_bytes"),
                        )
                        owner.snapshot = None
                    state["coupling_array"] = owner.array
                    state["_factorized_plan"] = None
                    return owner.array

        @coupling_array.setter
        def coupling_array(self, value):
            state = object.__getattribute__(self, "__dict__")
            while True:
                owner = state.get("_factorized_owner")
                if owner is None:
                    # The inherited dataclass constructor writes this field
                    # before the factory creates any latent owner.
                    state["coupling_array"] = value
                    return
                with owner.lock:
                    if state.get("_factorized_owner") is not owner:
                        continue
                    detached = _CoefficientOwner(value)
                    state["_factorized_owner"] = detached
                    state["_factorized_plan"] = None
                    state["_factorized_lock"] = detached.lock
                    state["coupling_array"] = value
                    return

        @coupling_array.deleter
        def coupling_array(self):
            state = object.__getattribute__(self, "__dict__")
            while True:
                owner = state.get("_factorized_owner")
                if owner is None:
                    if "coupling_array" not in state:
                        raise AttributeError("coupling_array")
                    del state["coupling_array"]
                    return
                with owner.lock:
                    if state.get("_factorized_owner") is not owner:
                        continue
                    if not owner.present:
                        raise AttributeError("coupling_array")
                    detached = _CoefficientOwner(None, present=False)
                    state["_factorized_owner"] = detached
                    state["_factorized_plan"] = None
                    state["_factorized_lock"] = detached.lock
                    state.pop("coupling_array", None)
                    return

        def __getstate__(self):
            state = object.__getattribute__(self, "__dict__")
            owner = state.get("_factorized_owner")
            if owner is None:
                return dict(state)
            with owner.lock:
                saved = dict(state)
                saved["_factorized_plan"] = owner.snapshot
                if owner.present:
                    saved["coupling_array"] = owner.array
                else:
                    saved.pop("coupling_array", None)
                saved.pop("_factorized_lock", None)
                return saved

        def __setstate__(self, state):
            object.__getattribute__(self, "__dict__").update(state)
            owner = state.get("_factorized_owner")
            if owner is not None:
                object.__setattr__(self, "_factorized_lock", owner.lock)

        def _static_values(
            self, c_batches, r_product, jacobians, *, r_batches=(), where=" (coupling)"
        ):
            if native_factorized_integrand(self):
                state = object.__getattribute__(self, "__dict__")
                while True:
                    owner = state["_factorized_owner"]
                    with owner.lock:
                        if state.get("_factorized_owner") is not owner:
                            continue
                        snapshot = owner.snapshot
                        if (
                            snapshot is not None
                            and self._contraction_backend == "factorized"
                            and self.diagram_term.propagator_indices
                            == snapshot.prop_idx
                        ):
                            values = _contract(
                                snapshot,
                                self.fixed_indices,
                                c_batches,
                                r_product,
                                jacobians,
                                r_batches,
                                self._e_psi,
                                self._contraction_max_bytes,
                            )
                            if values is not None:
                                return values
                        break
            return DiagramIntegrand._static_values(
                self, c_batches, r_product, jacobians, r_batches=r_batches, where=where
            )

    _FactorizedIntegrand.__qualname__ = "_FactorizedIntegrand"
    globals()["_FactorizedIntegrand"] = _FactorizedIntegrand
    _CLASS = _FactorizedIntegrand
    return _CLASS


def __getattr__(name):
    # Standard pickle imports the private class by name in a fresh worker.
    if name == "_FactorizedIntegrand":
        return _integrand_class()
    raise AttributeError(name)


def native_factorized_integrand(integrand) -> bool:
    """A prepared factorized integrand whose latent owner is still bound."""
    if _CLASS is None or type(integrand) is not _CLASS:
        return False
    state = object.__getattribute__(integrand, "__dict__")
    owner = state.get("_factorized_owner")
    return (
        type(owner) is _CoefficientOwner
        and owner.present
        and type(state.get("diagram_term")) is _pt.DiagramTerm
    )


def try_build_factorized(term, spatial, static_cv, fixed, max_bytes):
    """Build an admitted factorization, or check its dense fallback's budget.

    A known oversized coefficient fallback raises before allocating that array.
    Increasing the explicit factorized budget or choosing ordered construction
    remains the caller's choice; the old backends do not gain this new policy.
    """
    result = _try_build_factorized(term, spatial, static_cv, fixed, max_bytes)
    if result is None:
        check_materialization_budget(term, static_cv, max_bytes)
    return result


def _component(array):
    """``(component, phase, range)`` of a real or imaginary array."""
    if _native_dtype(array, np.float64):
        component, phase = array, 1 + 0j
    elif not np.any(array.real):
        component, phase = array.imag, 1j
    elif not np.any(array.imag):
        component, phase = array.real, 1 + 0j
    else:
        return None
    scanned = finite_range(component)
    if scanned is None:
        return None
    return component, phase, scanned


def _chunks(shape, limit):
    """Prefix length ``p`` and the prefix assignments of a chunked scan.

    The trailing ``shape[p:]`` holds at most ``limit`` elements; ``None`` when
    that takes more than 4096 chunks.
    """
    for p in range(len(shape) + 1):
        if math.prod(shape[p:]) <= limit:
            if math.prod(shape[:p]) > 4096:
                return None
            return p, np.ndindex(*shape[:p])
    return None


def _factor_sign(component):
    """1 or -1 when every nonzero entry has that sign, 0 when all are zero,
    ``None`` when the entries hold both signs (entries are finite)."""
    if component.size == 0:
        return 0
    high = float(np.max(component))
    low = float(np.min(component))
    if low >= 0.0:
        return 1 if high > 0.0 else 0
    if high <= 0.0:
        return -1
    return None


def _signs_agree(monomials, components):
    """Whether the factor signs alone prove that no two monomials differ in sign.

    A monomial whose factors each hold one sign (zeros allowed) is, at every
    assignment, zero or of one fixed sign: its phase's sign times its factor
    signs.  Returns ``True`` when every monomial that is not identically zero
    has the same fixed sign.  The work is one pass over each distinct factor.
    """
    signs = {}
    common = 0
    for monomial in monomials:
        if monomial.phase == 0.0:
            continue
        sign = 1 if monomial.phase > 0.0 else -1
        for slot, _, slices, _ in monomial.factors:
            # Slices are hashable only from Python 3.12.
            key = (slot, tuple(None if type(item) is slice else item for item in slices))
            if key not in signs:
                signs[key] = _factor_sign(components[slot][0][slices])
            if signs[key] is None:
                return False
            sign *= signs[key]
        if sign and common and sign != common:
            return False
        common = common or sign
    return True


def _nonzero_count(monomial, components, shape, available):
    """Assignments at which every factor of ``monomial`` is nonzero.

    Contracts the 0/1 patterns of the factors along a planned path, so the
    work is that of the contraction, not of the assignment tensor.  Returns
    0, a valid lower bound, when the patterns and the path do not fit
    ``available`` bytes.
    """
    if monomial.phase == 0.0:
        return 0
    operands = []
    used = set()
    for slot, _, slices, labels in monomial.factors:
        operands.append(((components[slot][0][slices] != 0).astype(np.float64), labels))
        used.update(labels)
    free = math.prod(dim for axis, dim in enumerate(shape, 1) if axis not in used)
    if not operands:
        return free
    # A one-element sample axis gives the planner its required output label.
    operands.append((np.ones(1), (0,)))
    plan = _dense.plan_contraction(
        tuple(array.shape for array, _ in operands),
        tuple(labels for _, labels in operands),
    )
    if plan is None:
        return 0
    steps, _, workspace = plan
    if 8 * (workspace + sum(array.size for array, _ in operands)) > available:
        return 0
    return free * int(_dense.execute(operands, steps)[0])


def _coefficient_pattern(monomials, components, shape, available):
    """Sign agreement and zero pattern of the summed coefficient.

    The sign of monomial ``t`` at assignment ``a`` is the sign of its real
    phase times the signs of its factor entries, computed exactly on int8
    sign arrays.  Returns ``(nonzero, mask)``: the number of assignments
    where some monomial is nonzero and, for at most ``_MASK_ELEMENTS``
    assignments, that pattern as a 0/1 float64 array.  When this
    per-assignment check would exceed ``_MAX_SIGN_WORK`` element operations
    or ``available`` bytes, agreement must follow from the factor signs
    alone (:func:`_signs_agree`); ``nonzero`` is then the largest count of
    one monomial, a lower bound that is exact for a single monomial, and
    ``mask`` is ``None``.  Returns ``None`` when two monomials have opposite
    signs at some assignment, or when neither check proves agreement.
    """
    assignments = math.prod(shape)
    terms = len(monomials)
    work = assignments * sum(len(m.factors) + 1 for m in monomials)
    keep_mask = assignments <= _MASK_ELEMENTS
    scan_bytes = available - sum(component.size for component, _, _ in components)
    scan_bytes -= 8 * assignments if keep_mask else 0
    chunked = _chunks(shape, min(_SIGN_CHUNK, scan_bytes // 4)) if scan_bytes >= 4 else None
    if work > _MAX_SIGN_WORK or chunked is None:
        if terms > 1 and not _signs_agree(monomials, components):
            return None
        nonzero = max(
            _nonzero_count(monomial, components, shape, available) for monomial in monomials
        )
        return nonzero, None
    p, prefixes = chunked
    signs = [np.sign(component).astype(np.int8) for component, _, _ in components]
    tail = list(range(p + 1, len(shape) + 1))
    ones = np.broadcast_to(np.ones((), dtype=np.int8), shape[p:])
    mask = np.zeros(shape, dtype=np.float64) if keep_mask else None
    nonzero = 0
    for prefix in prefixes:
        positive = np.zeros(shape[p:], dtype=bool)
        negative = np.zeros(shape[p:], dtype=bool)
        for monomial in monomials:
            if monomial.phase == 0.0:
                continue
            operands = [ones, tail]
            for slot, _, slices, labels in monomial.factors:
                index = tuple(
                    prefix[label - 1] if label <= p else slice(None) for label in labels
                )
                kept = [label for label in labels if label > p]
                operands += [signs[slot][slices][index], kept]
            value = np.einsum(*operands, tail)
            if monomial.phase < 0.0:
                positive |= value < 0
                negative |= value > 0
            else:
                positive |= value > 0
                negative |= value < 0
            # Stop at the first monomial that conflicts with an earlier one.
            if terms > 1 and np.any(positive & negative):
                return None
        present = positive | negative
        nonzero += int(np.count_nonzero(present))
        if mask is not None:
            mask[prefix] = present
    if mask is not None:
        mask.flags.writeable = False
    return nonzero, mask


def _lower(expression, static_cv, names, fixed):
    """Lower a coupling sum into its monomials.

    Returns ``(groups, used)``: each group lists one monomial's factors,
    ``("r", value)``, ``("i",)`` or ``("s", slot, slices, labels)``, and
    ``used`` maps coupling names to slots.  ``None`` for any unsupported node.
    """
    used = {}

    def lower(node, factors, depth):
        if depth > _MAX_DEPTH:
            return False
        kind = type(node)
        if kind is Rational:
            if (
                type(node.numerator) is not int
                or type(node.denominator) is not int
                or not node.denominator
            ):
                return False
            try:
                value = node.numerator / node.denominator
            except (OverflowError, ZeroDivisionError):
                return False
            if not math.isfinite(value):
                return False
            factors.append(("r", value))
            return True
        if kind is ImaginaryUnit:
            factors.append(("i",))
            return True
        if kind is Symbol:
            if (
                type(node.name) is not str
                or type(node.indices) is not tuple
                or node.name not in static_cv
                or any(not _ordinary_label(label) for label in node.indices)
            ):
                return False
            array = static_cv[node.name]
            if array.ndim != len(node.indices):
                return False
            slices = []
            labels = []
            for label, dim in zip(node.indices, array.shape):
                if label in names:
                    axis, size = names[label]
                    if dim != size:
                        return False
                    slices.append(slice(None))
                    labels.append(axis)
                elif label in fixed and 0 <= fixed[label] < dim:
                    slices.append(fixed[label])
                else:
                    return False
            slot = used.setdefault(node.name, len(used))
            factors.append(("s", slot, tuple(slices), tuple(labels)))
            return True
        if kind is not Product or type(node.factors) is not tuple:
            return False
        return all(lower(child, factors, depth + 1) for child in node.factors)

    if type(expression) is Sum:
        if type(expression.terms) is not tuple or not expression.terms:
            return None
        monomials = expression.terms
        if any(type(monomial) is not Product for monomial in monomials):
            return None
    else:
        monomials = (expression,)
    groups = []
    for monomial in monomials:
        factors = []
        if not lower(monomial, factors, 1) or len(factors) > _MAX_FACTORS:
            return None
        groups.append(tuple(factors))
    return groups, used


def _prefactor(term):
    """``(-i)**n_response`` times the rational prefactor, or ``None``."""
    rational = term.rational_prefactor
    n_response = term.n_response
    e_psi = term.n_external_response
    if (
        type(rational) is not Rational
        or type(rational.numerator) is not int
        or type(rational.denominator) is not int
        or not rational.denominator
        or type(n_response) is not int
        or not 0 <= n_response <= 1024
        or type(e_psi) is not int
        or not 0 <= e_psi <= 1024
        or (1j) ** e_psi not in QUARTER_TURNS
    ):
        return None
    try:
        prefactor = [1.0, -1j, -1.0, 1j][n_response % 4] * (
            rational.numerator / rational.denominator
        )
    except (OverflowError, ZeroDivisionError):
        return None
    if not math.isfinite(abs(prefactor)) or prefactor == 0:
        return None
    return prefactor


def _monomials(groups, components, prefactor, e_psi):
    """The lowered monomials with their real scalar phases, or ``None``.

    Each term must be real after the observable's phase on its own;
    imaginary terms are never admitted through cancellation against each
    other.
    """
    monomials = []
    for factors in groups:
        scalar = complex(prefactor) * (1j**e_psi)
        operands = []
        magnitudes = [abs(prefactor)]
        for factor in factors:
            if factor[0] == "s":
                _, phase, _ = components[factor[1]]
                scalar *= phase
                operands.append((factor[1], phase == 1j, factor[2], factor[3]))
            elif factor[0] == "r":
                scalar *= factor[1]
                magnitudes.append(abs(factor[1]))
            else:
                scalar *= 1j
        if scalar.imag != 0 or not math.isfinite(scalar.real):
            return None
        # A factor of magnitude 1 changes no exponent; zero has no smallest
        # nonzero magnitude, as in finite_range.
        scales = tuple(
            (value, value if value else math.inf) for value in magnitudes if value != 1.0
        )
        monomials.append(_Monomial(tuple(operands), scalar.real, scales))
    return monomials


def _try_build_factorized(term, spatial, static_cv, fixed, max_bytes):
    """Return a lazy factorized integrand, otherwise decline with ``None``.

    Callers select this helper only for an explicit factorized backend.
    Admission never evaluates a user callable or the symbolic coefficient
    evaluator.
    """
    if (
        type(max_bytes) is not int
        or max_bytes <= _SCAN_RESERVE
        or not errors_allow_backend()
        or not _native_term(term)
        or type(static_cv) is not dict
        or any(type(name) is not str for name in static_cv)
        or not fixed_labels_ok(fixed)
    ):
        return None
    for value in static_cv.values():
        if type(value) is not np.ndarray or not (
            _native_dtype(value, np.float64) or _native_dtype(value, np.complex128)
        ):
            return None
    summation = term.summation_indices
    if not 1 <= len(summation) <= 51 or len({name for name, _ in summation}) != len(
        summation
    ):
        return None
    names = {name: (axis, dim) for axis, (name, dim) in enumerate(summation, 1)}
    prop_idx = term.propagator_indices
    if prop_idx != summation or math.prod(dim for _, dim in prop_idx) <= 1:
        return None
    prefactor = _prefactor(term)
    if prefactor is None:
        return None
    e_psi = term.n_external_response
    lowered = _lower(term.coupling_sum, static_cv, names, fixed)
    if lowered is None:
        return None
    groups, used = lowered
    if not used:
        return None
    nbytes = sum(static_cv[name].nbytes for name in used)
    if nbytes + _SCAN_RESERVE > max_bytes:
        return None
    arrays = []
    components = []
    for name in used:
        owned = np.array(static_cv[name], copy=True, order="K")
        owned.flags.writeable = False
        component = _component(owned)
        if component is None:
            return None
        arrays.append(owned)
        components.append(component)
    ranges = tuple(component[2] for component in components)
    monomials = _monomials(groups, components, prefactor, e_psi)
    if monomials is None:
        return None
    pattern = _coefficient_pattern(
        monomials,
        components,
        tuple(dim for _, dim in prop_idx),
        max_bytes - nbytes - _SCAN_RESERVE,
    )
    if pattern is None:
        return None
    nonzero, mask = pattern
    if mask is not None:
        nbytes += mask.nbytes
    dtype = (
        complex
        if isinstance(prefactor, complex)
        or any(_native_dtype(array, np.complex128) for array in static_cv.values())
        else float
    )
    nbytes += np.dtype(dtype).itemsize  # still-live public 0-d placeholder
    if nbytes + _SCAN_RESERVE > max_bytes:
        return None
    snapshot = _Snapshot(
        tuple(used),
        tuple(arrays),
        ranges,
        tuple(monomials),
        prop_idx,
        tuple(fixed.items()),
        dtype,
        e_psi,
        nbytes,
        nonzero,
        mask,
    )
    cls = _integrand_class()
    integrand = cls(
        diagram_term=term,
        spatial=spatial,
        coupling_array=np.zeros((), dtype=dtype),
        fixed_indices=dict(fixed),
        _contraction_backend="factorized",
        _contraction_max_bytes=max_bytes,
    )
    integrand_state = object.__getattribute__(integrand, "__dict__")
    owner = _CoefficientOwner(integrand_state["coupling_array"], snapshot)
    integrand_state["_factorized_owner"] = owner
    integrand_state["_factorized_plan"] = snapshot
    integrand_state["_factorized_lock"] = owner.lock
    return integrand
