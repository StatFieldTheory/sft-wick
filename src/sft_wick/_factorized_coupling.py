"""The ``"factorized"`` backend: contract the original coupling factors.

Preparation keeps owned copies of the static coupling tensors of a native
monomial or sum of monomials instead of the dense coefficient over all
propagator component indices.  Each call contracts every monomial with the
propagator batches and adds the monomial results, under the accuracy and
behaviour contract of :mod:`sft_wick._contraction_admission`.  When a call
declines, or when the public ``coupling_array`` is read, the ordered
preparation (``DiagramTerm.evaluate_coupling``) builds the dense coefficient
from the owned copies and the integrand continues on the ordered fold.

A sum of monomials is prepared only when no two monomials have opposite
signs at any component assignment.  Monomials that cancel would leave the
contracted result with an error proportional to the sum of the monomial
magnitudes, while the ordered fold sums the monomials into one coefficient
before it multiplies the propagators.  The sign check runs once, on sign
arrays of the factors, and also counts the nonzero coefficients for the
cost model (:func:`sft_wick._factorized_lowering.coefficient_pattern`).
When that per-assignment check would take too much work, the sum is
prepared only if the factor signs alone prove agreement (every factor holds
one sign apart from zeros, and every monomial has the same resulting sign),
and the nonzero count is replaced by a lower bound.

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
from ._contraction_admission import (
    _native_dtype,
    _ordinary_label,
    call_inputs_ok,
    coupling_array_ok,
    errors_allow_backend,
    factorized_overhead,
    factorized_workspace,
    fixed_labels_ok,
    label_axes,
    label_item_ok,
    minimum_backend_cost,
    ordered_cost,
    scalar_range,
)
from ._coupling_plan import _CouplingPlan
from ._dense_contraction import finish, live_assignments, live_cost_admits, plan_call
from ._factorized_lowering import (
    coefficient_pattern,
    component,
    lower_coupling,
    prefactor,
)
from .expressions import _NATIVE_NUMERIC_EXPRESSION_CLASSES

# Definition-time class binding, as the ordered numerical evaluators use.
Propagator = _NATIVE_NUMERIC_EXPRESSION_CLASSES[-1]

#: Payload reserved beside a materialized coefficient.
_SCAN_RESERVE = 64 * 4096
#: Element operations the build-time per-assignment sign check may spend
#: (monomials times assignments times factors).  Above it a sum is prepared
#: only when the signs of its factors prove agreement.
_MAX_SIGN_WORK = 2**27

_CLASS = None
_CLASS_LOCK = Lock()
#: The ordered integrand class, bound when the factorized class is created.
_ORDERED = None


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
    #: :func:`~sft_wick._contraction_admission.finite_range` of each array's
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
        and all(label_item_ok(item) for item in summation)
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


def check_materialization_budget(term, max_bytes):
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


# ------------------------------------------------------------- one call


def _contract(snapshot, fixed, cb, rp, jac, rb, e_psi, max_bytes):
    """Contract every monomial with the propagator batches, or decline."""
    # The budget and the underflow policy come first, as in ordered calls:
    # an unusable budget declines before e_psi is compared.
    if (
        type(max_bytes) is not int
        or max_bytes <= 0
        or not errors_allow_backend()
        or e_psi != snapshot.e_psi
        or not call_inputs_ok(max_bytes, fixed, cb, rb, rp)
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
    mask_shape = None if snapshot.mask is None else snapshot.mask.shape
    planned = plan_call(
        _groups(snapshot), (cb, rb, rp, jac), names, fixed, dims, overhead, upper, mask_shape
    )
    if planned is None:
        return None
    if factorized_workspace(planned.contraction, planned.layout, planned.counting) > max_bytes:
        return None
    live, counted = live_assignments(dims, snapshot.mask, (), restricted=planned.restricted)
    if live is None:
        return None
    if not live_cost_admits(planned, assignments, nonzero, min(live, nonzero), counted, samples):
        return None
    ranges = tuple(
        tuple(snapshot.ranges[slot] for slot, _, _, _ in m.factors) + m.scales
        for m in snapshot.monomials
    )
    return finish(planned.contraction, planned.layout, ranges, assignments, phase_is_factor=False)


def _groups(snapshot):
    """``(phase, operands)`` of every monomial: views of its owned factors."""
    groups = []
    for monomial in snapshot.monomials:
        operands = []
        for slot, imaginary, slices, labels in monomial.factors:
            array = snapshot.arrays[slot]
            part = array.imag if imaginary else array.real
            operands.append((part[slices], labels))
        groups.append((monomial.phase, tuple(operands)))
    return tuple(groups)


# ----------------------------------------------------- integrand class


def _read_coupling(self):
    """``coupling_array``: materializes the coefficient on first read."""
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


def _write_coupling(self, value):
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


def _delete_coupling(self):
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


def _get_state(self):
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


def _set_state(self, state):
    object.__getattribute__(self, "__dict__").update(state)
    owner = state.get("_factorized_owner")
    if owner is not None:
        object.__setattr__(self, "_factorized_lock", owner.lock)


def _contract_or_fold(
    self, c_batches, r_product, jacobians, *, r_batches=(), where=" (coupling)"
):
    """``_static_values`` of a factorized integrand: the contraction when it
    is admitted, otherwise the ordered fold (which materializes)."""
    if native_factorized_integrand(self):
        values = _contract_with_owner(self, c_batches, r_product, jacobians, r_batches)
        if values is not None:
            return values
    return _ORDERED._static_values(
        self, c_batches, r_product, jacobians, r_batches=r_batches, where=where
    )


def _contract_with_owner(self, c_batches, r_product, jacobians, r_batches):
    """:func:`_contract` on the owner's snapshot, under its lock."""
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
                and self.diagram_term.propagator_indices == snapshot.prop_idx
            ):
                return _contract(
                    snapshot,
                    self.fixed_indices,
                    c_batches,
                    r_product,
                    jacobians,
                    r_batches,
                    self._e_psi,
                    self._contraction_max_bytes,
                )
            return None


def _integrand_class():
    if _CLASS is not None:
        return _CLASS
    with _CLASS_LOCK:
        return _create_integrand_class()


def _create_integrand_class():
    global _CLASS, _ORDERED
    if _CLASS is not None:
        return _CLASS
    from .evaluate import DiagramIntegrand

    class _FactorizedIntegrand(DiagramIntegrand):
        coupling_array = property(_read_coupling, _write_coupling, _delete_coupling)
        __getstate__ = _get_state
        __setstate__ = _set_state
        _static_values = _contract_or_fold

    _FactorizedIntegrand.__qualname__ = "_FactorizedIntegrand"
    globals()["_FactorizedIntegrand"] = _FactorizedIntegrand
    _ORDERED = DiagramIntegrand
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


# --------------------------------------------------------------- build


def try_build_factorized(term, spatial, static_cv, fixed, max_bytes):
    """Build an admitted factorization, or check its dense fallback's budget.

    A known oversized coefficient fallback raises before allocating that array.
    Increasing the explicit factorized budget or choosing ordered construction
    remains the caller's choice; the old backends do not gain this new policy.
    """
    result = _try_build_factorized(term, spatial, static_cv, fixed, max_bytes)
    if result is None:
        check_materialization_budget(term, max_bytes)
    return result


def _try_build_factorized(term, spatial, static_cv, fixed, max_bytes):
    """Return a lazy factorized integrand, otherwise decline with ``None``.

    Callers select this helper only for an explicit factorized backend.
    Admission never evaluates a user callable or the symbolic coefficient
    evaluator.
    """
    if not _build_inputs_ok(term, static_cv, fixed, max_bytes):
        return None
    lowered = _lowered_term(term, static_cv, fixed)
    if lowered is None:
        return None
    snapshot = _take_snapshot(static_cv, fixed, max_bytes, *lowered)
    if snapshot is None:
        return None
    integrand = _integrand_class()(
        diagram_term=term,
        spatial=spatial,
        coupling_array=np.zeros((), dtype=snapshot.dtype),
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


def _build_inputs_ok(term, static_cv, fixed, max_bytes) -> bool:
    """A budget above the scan reserve, the underflow policy, a native term,
    a dict of float64 or complex128 arrays and identifier fixed labels."""
    return (
        type(max_bytes) is int
        and max_bytes > _SCAN_RESERVE
        and errors_allow_backend()
        and _native_term(term)
        and type(static_cv) is dict
        and all(type(name) is str for name in static_cv)
        and fixed_labels_ok(fixed)
        and all(coupling_array_ok(value) for value in static_cv.values())
    )


def _lowered_term(term, static_cv, fixed):
    """``(prop_idx, prefactor, e_psi, groups, used, dtype)`` of a term whose
    coupling lowers into monomials of static tensors, or ``None``."""
    summation = term.summation_indices
    names = label_axes(summation)
    if names is None:
        return None
    prop_idx = term.propagator_indices
    if prop_idx != summation or math.prod(dim for _, dim in prop_idx) <= 1:
        return None
    scalar = prefactor(term)
    if scalar is None:
        return None
    e_psi = term.n_external_response
    lowered = lower_coupling(term.coupling_sum, static_cv, names, fixed)
    if lowered is None:
        return None
    groups, used = lowered
    if not used:
        return None
    dtype = (
        complex
        if isinstance(scalar, complex)
        or any(_native_dtype(array, np.complex128) for array in static_cv.values())
        else float
    )
    # The ordered preparation converts each monomial's value to this dtype;
    # a factor ``i`` makes the value complex, which ``float`` rejects with
    # TypeError, so such a coupling keeps the ordered preparation.
    if dtype is float and any(factor[0] == "i" for factors in groups for factor in factors):
        return None
    return prop_idx, scalar, e_psi, groups, used, dtype


def _take_snapshot(static_cv, fixed, max_bytes, prop_idx, scalar, e_psi, groups, used, dtype):
    """Owned copies of the used tensors, the monomials and the coefficient
    pattern, within the budget; ``None`` otherwise."""
    nbytes = sum(static_cv[name].nbytes for name in used)
    if nbytes + _SCAN_RESERVE > max_bytes:
        return None
    arrays = []
    components = []
    for name in used:
        owned = np.array(static_cv[name], copy=True, order="K")
        owned.flags.writeable = False
        part = component(owned)
        if part is None:
            return None
        arrays.append(owned)
        components.append(part)
    monomials = _monomials(groups, components, scalar, e_psi)
    if monomials is None:
        return None
    pattern = coefficient_pattern(
        monomials,
        components,
        tuple(dim for _, dim in prop_idx),
        max_bytes - nbytes - _SCAN_RESERVE,
        _MAX_SIGN_WORK,
    )
    if pattern is None:
        return None
    nonzero, mask = pattern
    if mask is not None:
        nbytes += mask.nbytes
    nbytes += np.dtype(dtype).itemsize  # still-live public 0-d placeholder
    if nbytes + _SCAN_RESERVE > max_bytes:
        return None
    ranges = tuple(part[2] for part in components)
    return _Snapshot(
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


def _monomials(groups, components, scalar, e_psi):
    """The lowered monomials with their real scalar phases, or ``None``.

    Each term must be real after the observable's phase on its own;
    imaginary terms are never admitted through cancellation against each
    other.
    """
    monomials = []
    for factors in groups:
        phase_value = complex(scalar) * (1j**e_psi)
        operands = []
        magnitudes = [abs(scalar)]
        for factor in factors:
            if factor[0] == "s":
                _, phase, _ = components[factor[1]]
                phase_value *= phase
                operands.append((factor[1], phase == 1j, factor[2], factor[3]))
            elif factor[0] == "r":
                phase_value *= factor[1]
                magnitudes.append(abs(factor[1]))
            else:
                phase_value *= 1j
        if phase_value.imag != 0 or not math.isfinite(phase_value.real):
            return None
        # A factor of magnitude 1 changes no exponent; zero has no smallest
        # nonzero magnitude, as in finite_range.
        scales = tuple(scalar_range(value) for value in magnitudes if value != 1.0)
        monomials.append(_Monomial(tuple(operands), phase_value.real, scales))
    return monomials
