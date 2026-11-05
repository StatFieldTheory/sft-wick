"""Numerical evaluation pipeline for DiagramTerm objects.

Provides the 4-step workflow:
1. Coupling coefficients: DiagramTerm.evaluate_coupling() (existing)
2. Spatial structure analysis: analyze_spatial()
3. Propagator evaluation: PropagatorModel + PropagatorCache
4. Contraction and integration: DiagramIntegrand
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import cached_property, lru_cache
from math import fsum, isfinite
from types import MethodType, SimpleNamespace
from typing import Any, Callable, Iterable, NamedTuple

import hashlib
import inspect
import itertools

import sys
import warnings

import numpy as np

from .expressions import Propagator
from .indices import is_literal_component, literal_component, pinned_component
from .wick import r_vanishes
from ._propagator_lookup import (
    _PropagatorLookupCache,
    validate_cache_options,
    wrap_propagator_cache,
)
from ._dense_contraction import (
    DEFAULT_CONTRACTION_MAX_BYTES,
    try_static_contraction,
    validate_contraction_options,
)

# ---------------------------------------------------------------------------
# Step 2: Spatial Structure Analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpatialStructure:
    """Result of analyzing R-propagator connectivity and time orderings.

    R propagators of the form ``δ(n−n') Θ(t−t') R_time(t,t')`` identify
    directions along R-chains and impose causal time orderings.
    """

    #: Groups of spatial points connected by R propagators (share direction).
    direction_groups: tuple[frozenset[str], ...]

    #: spatial_point → representative direction variable name (e.g. ``'n_x'``).
    direction_map: dict[str, str]

    #: Time ordering pairs: ``(earlier_point, later_point)`` from R causality.
    #: ``R(a, b)`` with ``Θ(t_a − t_b)`` means ``t_b ≤ t_a``.
    time_orderings: tuple[tuple[str, str], ...]

    #: R propagators as ``(spatial_left, spatial_right)`` pairs.
    r_propagators: tuple[tuple[str, str], ...]

    #: C propagators: ``(spatial_left, spatial_right, index_left, index_right)``.
    c_propagators: tuple[tuple[str, str, str | None, str | None], ...]

    #: Time integration variables, topologically sorted (innermost first).
    #: For an ``equal_time`` non-local vertex, only the canonical
    #: representative leg appears here; the other (m-1) legs of that
    #: vertex are removed and their time values are read through the
    #: ``equal_time_aliases`` map below.
    time_integration_vars: tuple[str, ...]

    #: Surviving direction integration variables (one per R-component that
    #: contains only integration points; typically empty).
    direction_integration_vars: tuple[str, ...]

    #: External (non-integration) spatial points.
    external_points: tuple[str, ...]

    #: Maps non-representative internal spatial labels to their canonical
    #: time representative, propagated from
    #: ``DiagramTerm.equal_time_aliases``. The integration loops look up
    #: every leg's time via ``equal_time_aliases.get(label, label)`` so
    #: aliased legs share a single integration variable while keeping
    #: independent spatial labels. Empty tuple ⇒ no aliasing (original
    #: cross-spacetime cumulant behaviour).
    equal_time_aliases: tuple[tuple[str, str], ...] = ()

    #: ``(partner_label, leg_label)`` pairs identifying R-propagators
    #: whose factor has been absorbed into an upstream
    #: ``NonLocalVertex(already_R_contracted=True)`` callable.
    #: Propagated straight from ``DiagramTerm.r_absorbed_pairs``. The
    #: integrand R-product loop iterates over this set to **skip** the
    #: matching propagators (so the factor becomes 1 rather than
    #: ``R(t, t) = 0`` under Itô after the leg time aliases onto the
    #: partner's). The leg's time / direction collapse onto the
    #: partner's via the accompanying ``equal_time_aliases`` entries.
    r_absorbed_pairs: tuple[tuple[str, str], ...] = ()


def _kept_r_propagators(
    spatial: "SpatialStructure",
) -> tuple[tuple[str, str], ...]:
    """Return ``spatial.r_propagators`` minus any pair listed in
    ``spatial.r_absorbed_pairs``.

    Absorbed R-propagators contribute to direction grouping and time
    ordering (the leg label is union-find'd with its partner via the
    original R-propagator) but their R-factor has been folded into an
    upstream ``κ^(m)_R`` callable. The integrand R-product loops must
    skip them; calling this helper centralises the filter so every
    factor-multiplication site applies the same rule.
    """
    if not spatial.r_absorbed_pairs:
        return spatial.r_propagators
    absorbed = set(spatial.r_absorbed_pairs)
    return tuple(p for p in spatial.r_propagators if p not in absorbed)


#: ``ValueError`` text for each kind of :func:`_r_index_mismatch`.
_R_MISMATCH = {
    "scalar": (
        "{r} carries two different component indices, but the propagator "
        "cache's R is a scalar (cache.model.iso_R is True) and is multiplied "
        "in without their Kronecker delta, so the leg components would be "
        "summed instead of pinned.  Build the terms with "
        "compute_moment(..., iso_R=True) or diag_R=True; at L1, do not pass "
        "iso_R=False together with diag_R=False to System.expand."
    ),
    "matrix": (
        "{r} has no component index on one or both legs, which is how "
        "compute_moment(..., iso_R=True) writes a scalar R, but the "
        "propagator cache's R is a matrix (cache.model.iso_R is False), and "
        "an index-free R is evaluated as its trace: N times too large for R "
        "proportional to the N x N identity.  Build the terms with "
        "iso_R=False (diag_R=True keeps R diagonal), or use a cache whose R "
        "is a scalar."
    ),
    "absorbed": (
        "{r} is absorbed into an already_R_contracted vertex, so its factor "
        "is skipped, and what stands in for it is the Kronecker delta of its "
        "two component indices: the leg must carry its partner's index.  "
        "compute_moment writes that index on every absorbed leg; these terms "
        "were built another way (by hand, or by an earlier compute_moment, "
        "for instance loaded from an on-disk expansion cache), and "
        "evaluating them would sum the leg components instead of pinning "
        "them.  Rebuild them with compute_moment or System.expand."
    ),
}


def _retarded(t_left: Any, t_right: Any) -> Any:
    """Θ of an R propagator, on scalars or arrays: ``t_left > t_right``.  R is
    retarded, and strict by Itô, so equal times give 0 and so does a NaN."""
    return t_left > t_right


def _r_index_mismatch(dt: "DiagramTerm", iso_R: bool) -> str | None:
    """Why ``dt``'s R propagators cannot be evaluated against a scalar R
    (``iso_R``) or a matrix R, or ``None`` if they can.

    ``compute_moment`` writes an R propagator with two component indices
    (the default), one repeated index (``diag_R=True``) or none
    (``iso_R=True``).  The evaluators read those indices only for a matrix
    R, and they skip the factor of an R absorbed into an
    ``already_R_contracted`` vertex.  So a scalar R, ``R_ab = delta_ab R``,
    needs its delta applied to the terms; a matrix R needs the indices; and
    an absorbed R needs the delta whatever the cache holds.  Each of the
    three pairings that break this used to return a wrong number without an
    error.
    """
    absorbed = set(getattr(dt, "r_absorbed_pairs", ()) or ())
    for p in dt.propagators:
        if p.kind != "R":
            continue
        il, ir = p.index_left, p.index_right
        two = il is not None and ir is not None and il != ir
        if (p.spatial_left, p.spatial_right) in absorbed:
            kind = "absorbed" if two else None
        elif iso_R:
            kind = "scalar" if two else None
        else:
            kind = "matrix" if il is None or ir is None else None
        if kind:
            return _R_MISMATCH[kind].format(
                r=f"R({p.spatial_left}, {p.spatial_right}) with component "
                f"indices ({il}, {ir})"
            )
    return None


def _cache_r_type(cache: Any) -> bool | None:
    """``cache.model.iso_R``, or ``None`` when :func:`_r_index_mismatch` does
    not apply: a cache that declares no R type (a test double), or a single
    component, for which the delta is 1 and the trace is R's only entry."""
    model = getattr(cache, "model", None)
    iso_R = getattr(model, "iso_R", None)
    if iso_R is None or getattr(model, "n_components", None) == 1:
        return None
    return bool(iso_R)


def _select_C_batch(
    C_batch: np.ndarray,
    a: int | None,
    b: int | None,
) -> np.ndarray:
    """Select a per-sample C component from either diagonal or full batches."""
    C_arr = np.asarray(C_batch)
    if C_arr.ndim == 3:
        if a is not None and b is not None:
            return C_arr[:, a, b]
        return np.einsum("iaa->i", C_arr)
    if C_arr.ndim != 2:
        raise ValueError(
            f"C batch must have shape (n, N) or (n, N, N); got {C_arr.shape}."
        )
    if a is not None and b is not None:
        if a != b:
            return np.zeros(C_arr.shape[0], dtype=C_arr.dtype)
        return C_arr[:, a]
    return C_arr.sum(axis=1)


#: Accepted ``c_method`` values and homogeneity modes, and the default grid
#: and node count of a C table.  The cache, :class:`Propagators`, the YAML
#: parser and the workflow schema read them from here.
C_METHODS = ("auto", "dblquad", "gauss_legendre")
HOMOGENEITIES = ("translation", "rotation", "general")
DEFAULT_N_GRID_T = 60
DEFAULT_C_N_GAUSS = 20

#: Significant digits of a lazy table's key.  A key relative to the value tells
#: small separations apart in any units.
LAZY_KEY_DIGITS = 13

#: Relative departure above which two component values are different: the
#: spread of the diagonal or an off-diagonal entry of an index-free C, and
#: the spread of the rates of a :class:`~sft_wick.workflow.DiagonalA`.  It
#: has no absolute floor, so small values are told apart like large ones.
_ISO_C_RTOL = 1e-9


def _departs_relative(values: Any, reference: Any) -> Any:
    """Where ``values`` differ from ``reference`` by more than
    :data:`_ISO_C_RTOL` of ``reference``.  A NaN does not count as a
    departure; callers that must refuse one test it themselves."""
    return np.abs(values - reference) > _ISO_C_RTOL * np.abs(reference)


def _isotropic_C(C: Any, *, batched: bool = False) -> Any:
    """The value ``c`` an index-free C propagator stands for.

    ``compute_moment(..., iso_C=True)`` strips the equal component indices
    from a C propagator and keeps any index sum in the coupling, so the
    factor left for the propagator is ``c`` in ``C_ab = δ_ab c``.  Up to
    0.4.2 every evaluator multiplied by the trace ``N c`` instead, N times
    too large per C propagator.

    Each point is judged against its own diagonal, so the verdict does not
    depend on which samples share a batch.

    Args:
        C: ``(N,)`` diagonal or ``(N, N)`` matrix at one point; with
            ``batched=True``, ``(n, N)`` or ``(n, N, N)``.

    Returns:
        ``c``: a scalar, or ``(n,)`` with ``batched=True``.

    Raises:
        ValueError: if C is not a multiple of the identity, so that no
            single ``c`` exists.
    """
    arr = np.asarray(C)
    if not batched:
        arr = arr[None, ...]
    if arr.ndim == 3:
        diag = np.diagonal(arr, axis1=1, axis2=2)
        off = np.abs(arr - diag[:, :, None] * np.eye(arr.shape[1]))
    elif arr.ndim == 2:
        diag, off = arr, None
    else:
        raise ValueError(
            f"C must have shape (N,) or (N, N) per point; got "
            f"{np.asarray(C).shape} (batched={batched})."
        )
    c = diag.mean(axis=1)
    if np.any(_departs_relative(diag, c[:, None])) or (
        off is not None and np.any(off > _ISO_C_RTOL * np.abs(c)[:, None, None])
    ):
        raise ValueError(
            "an index-free C propagator (compute_moment(..., iso_C=True)) "
            "stands for C_ab = δ_ab c, but this C is not a multiple of the "
            f"identity to {_ISO_C_RTOL:g} relative.  Build the terms with "
            "iso_C=False."
        )
    return c if batched else c[0]


def _topological_sort_times(
    integration_vars: tuple[str, ...],
    time_orderings: list[tuple[str, str]],
) -> list[str]:
    """Topological sort of integration time variables using causal ordering.

    Returns variables ordered so that the *innermost* integration variable
    (the one with the tightest bounds) comes first.  This is the reverse
    of the DAG order: leaves (earliest times) first.
    """
    # Build adjacency: earlier → later
    adj: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = {v: 0 for v in integration_vars}
    int_set = set(integration_vars)

    for earlier, later in time_orderings:
        if earlier in int_set and later in int_set:
            adj[later].append(earlier)
            in_degree[earlier] = in_degree.get(earlier, 0) + 1

    # Kahn's algorithm: start from nodes with in_degree 0 (latest times)
    queue = [v for v in integration_vars if in_degree[v] == 0]
    result: list[str] = []
    while queue:
        queue.sort()  # deterministic tie-breaking
        node = queue.pop(0)
        result.append(node)
        for neighbor in adj.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # result is latest-first; reverse to get innermost (earliest) first
    result.reverse()
    return result


def _declared(fn: Any, attr: str) -> Any:
    """The value ``fn`` sets for ``attr``, looking through ``__wrapped__``
    when ``fn`` itself does not set it (the MSR factor wrapper
    :attr:`NonLocalVertex.msr_coupling` returns, or a ``functools.wraps``
    wrapper).  ``None`` when nothing in the chain sets it."""
    seen: set[int] = set()
    while fn is not None and id(fn) not in seen:
        seen.add(id(fn))
        value = getattr(fn, attr, None)
        if value is not None:
            return value
        fn = getattr(fn, "__wrapped__", None)
    return None


def _declares(fn: Any, attr: str) -> bool:
    """Whether the callable ``fn`` sets ``attr`` to a true value (see
    :func:`_declared`)."""
    return bool(_declared(fn, attr))


def _closed_form_c(cache: Any) -> Any:
    """The callable that replaces the C quadrature of ``cache``, or ``None``."""
    return getattr(cache, "c_value_fn", None)


def _closed_form_value(value: Any, t1: Any, t2: Any) -> np.ndarray:
    """What a closed-form C returned, as a real array.

    A vectorised closed form called at a scalar time returns a batch of one,
    which loses its leading axis; a length-1 time array keeps it for the
    batched lookups.

    Raises:
        ValueError: for a complex array.  ``np.asarray(z, dtype=float)``
            would drop the imaginary part with only a ComplexWarning.
    """
    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        raise ValueError(
            "a closed-form C must be real; got a complex array.  C is a "
            "covariance of a real field."
        )
    arr = arr.astype(float, copy=False)
    if arr.ndim == 3 and arr.shape[0] == 1 and np.ndim(t1) == 0 and np.ndim(t2) == 0:
        return arr[0]
    return arr


def _declared_c_kink(cache: Any) -> bool | None:
    """What the closed-form C of ``cache``, or else ``cache`` itself, says in
    ``has_diagonal_kink``.  ``None`` when neither says anything."""
    flag = _declared(_closed_form_c(cache), "has_diagonal_kink")
    if flag is None:
        flag = _declared(cache, "has_diagonal_kink")
    return None if flag is None else bool(flag)


def _c_has_diagonal_kink(
    cache: Any,
    times: tuple[float, float] | None = None,
    positions: tuple = (),
) -> bool:
    """Whether ``C(t1, t2)`` has a derivative jump on ``t1 = t2``.

    The first of these that applies decides:

    * a **declaration**: ``has_diagonal_kink`` on a closed-form C
      (``cache.c_value_fn``) or on a cache that computes its own C
      (:class:`~sft_wick.spectral.SpectralPropagatorCache`).  ``False`` is
      trusted as ``True`` is;
    * **white noise**: ``C = ∫_{t_min}^{min(t1, t2)} R σ² R``, whose first
      derivative jumps by ``σ² R`` across the diagonal.  An undeclared
      closed form on such a model is split too: a time-dependent σ² can
      keep the jump to a part of the range the probe does not sample;
    * an **undeclared closed-form C**, which replaces the κ² of the model,
      is probed (:func:`_probe_c_diagonal_kink`) on
      ``times = (t_min, t_max)`` at each ``(x1, x2)`` of ``positions``.
      The integrators pass the range and the points of the diagram
      (:meth:`DiagramIntegrand._c_kink_at`);
    * a **κ² with a ``|Δt|`` cusp** (:func:`_kappa2_kinks_c`), read at each
      ``(x1, x2)`` of ``positions`` as well, since a kernel can be smooth or
      vanish at the origin.

    The quadrature tables interpolate C rather than reproducing it, but the
    kink is in the tabulated values and the split still pays: demo 8
    measured order-2 ``⟨φ_0 φ_1⟩`` on a table at 1.4e-05 / 1.7e-06 (12 / 20
    nodes) with a damped-cosine κ² and 9.5e-07 / 1.5e-07 with the kink
    declared, where QMC sits at 1.9e-07.
    """
    declared = _declared_c_kink(cache)
    if declared is not None:
        return declared
    if _has_white_noise(cache):
        return True
    if _closed_form_c(cache) is not None:
        return _closed_form_c_kinks(cache, times, positions)
    return _kappa2_kinks_c(cache, positions, times)


def _has_white_noise(cache: Any) -> bool:
    """Whether the model of ``cache`` carries a white-noise ``σ²``."""
    return getattr(getattr(cache, "model", None), "sigma2", None) is not None


def _c_kink_is_probed(cache: Any) -> bool:
    """Whether :func:`_c_has_diagonal_kink` reaches the closed-form probe."""
    return (
        _declared_c_kink(cache) is None
        and not _has_white_noise(cache)
        and _closed_form_c(cache) is not None
    )


def _kappa2_kinks_c(
    cache: Any, positions: tuple = (), times: tuple[float, float] | None = None
) -> bool:
    """Whether the model's ``κ²`` leaves a kink on C's time diagonal.

    ``C = ∫∫ R κ² R`` over ``[t_min, t1] × [t_min, t2]``, so
    ``∂²C/∂t1∂t2 = R(t1, t1) κ²(t1, t2) R(t2, t2)``: a kernel smooth away
    from ``λ1 = λ2`` and cusped there (``λ e^{−|Δt|/σ_t}``, a damped cosine
    ``e^{−|Δt|/ℓ} cos(kΔt)``, the exponential-pulse cumulants of demos 3-4)
    puts a jump in C's **third** derivative on ``t1 = t2``, where white
    noise puts one in the first.  A Gaussian kernel, or any kernel
    differentiable at ``Δt = 0``, leaves C smooth and is kept out of the
    split.  A kernel much narrower than the spacing of the nodes is smooth
    only in name: its C reads as kinked to the rule and is not split (see
    :class:`~sft_wick.workflow.specs.GaussianTemporal`).

    The question is the one ``PropagatorCache._kappa2_has_diagonal_cusp``
    already answers for the C quadrature itself: the built-in kernels
    declare ``has_diagonal_cusp``, and any other callable is probed from
    one-sided differences at two step sizes.  This defers to it.  A
    cache with no κ² (a custom cache) answers False.
    """
    if getattr(getattr(cache, "model", None), "kappa2", None) is None:
        return False
    probe = getattr(cache, "_kappa2_has_diagonal_cusp", None)
    if not callable(probe):
        return False
    span = None if times is None else times[1] - times[0]
    try:
        if not positions:
            return bool(probe(span=span))
        return any(probe(pair, span) for pair in positions)
    except Exception:
        return False


#: Where :func:`_probe_c_diagonal_kink` samples ``t1``, as fractions of the
#: range, and its steps in ``t2``, as fractions of the range: a factor 4
#: apart, from a size that resolves the range down to where a ``|Δt|`` cusp
#: of relative size 1e-6 still stands above rounding.  The early times see a
#: cusp that dies out with time-limited noise, the late ones one that sets
#: in late.
_C_KINK_PROBE_TIMES = (0.05, 0.15, 0.37, 0.71, 0.93)
_C_KINK_PROBE_STEPS = tuple(0.02 * 4.0**-k for k in range(12))
#: ``(p, weights on C(t, t + k h), k = -2..2)``: the difference that isolates
#: a jump in the ``p``-th derivative of C across ``h = 0``.  It is ``O(h^p)``
#: on such a jump and ``O(h^(p+1))`` on a smooth function.
_C_KINK_STENCILS = (
    (1, (0.0, 1.0, -2.0, 1.0, 0.0)),
    (2, (-1.0, 2.0, 0.0, -2.0, 1.0)),
    (3, (1.0, -4.0, 6.0, -4.0, 1.0)),
)
#: A rung pair counts as level when the ratio lies in this band; it would be
#: 1 on a jump, 1/4 on a smooth function and above 1 once rounding dominates.
_C_KINK_LEVEL_BAND = (0.6, 1.7)
#: Nodes per piece of the rule in :func:`_cut_moves_c_integral`, and the
#: relative change of the integral that counts: above rounding and above the
#: noise of a C computed by quadrature to the usual tolerances.
_C_CUT_PROBE_NODES = 32
_C_CUT_PROBE_TOL = 1e-10
#: Position pairs remembered per cache by :func:`_closed_form_c_kinks`.
_C_KINK_MEMO_SIZE = 1024


def _closed_form_c_kinks(
    cache: Any, times: tuple[float, float] | None, positions: tuple
) -> bool:
    """:func:`_probe_c_diagonal_kink` over ``positions``, once per range and
    pair of positions on one cache object.  A range and a pair have their own
    answer, because a feature can lie past a shorter range: a sweep pays per
    ``t_final`` and pair, and a sweep with ``n_jobs > 1`` per grid point,
    since each task unpickles its own cache.  A callable that declares
    ``has_diagonal_kink`` is not probed."""
    if times is None:
        t_min = float(getattr(getattr(cache, "model", None), "t_min", 0.0))
        times = (t_min, t_min + 3.0)
    if not positions:
        probe = getattr(cache, "_probe_positions", None)
        positions = (probe() if callable(probe) else (0.0, 0.0),)
    memo = getattr(cache, "_c_kink_memo", None)
    if memo is None or len(memo) >= _C_KINK_MEMO_SIZE:
        memo = cache._c_kink_memo = {}
    for x1, x2 in positions:
        key = (
            float(times[0]),
            float(times[1]),
            np.asarray(x1, dtype=float).tobytes(),
            np.asarray(x2, dtype=float).tobytes(),
        )
        if key not in memo:
            memo[key] = _probe_c_diagonal_kink(cache, times, x1, x2)
        if memo[key]:
            return True
    return False


def _probe_c_diagonal_kink(
    cache: Any, times: tuple[float, float], x1: Any, x2: Any
) -> bool:
    """Numerically decide whether a closed-form C jumps in a derivative of
    order 1, 2 or 3 across ``t1 = t2``.

    Order 1 is a ``|t1 − t2|`` cusp, the white-noise case.  Order 3 is what
    a κ² cusp leaves (:func:`_kappa2_kinks_c`), so a hand-written C of an
    exponential kernel is found too.

    A smooth C that is narrower across the diagonal than a Gauss-Legendre
    rule can resolve integrates like a kinked one, so it counts as well, at
    any width, and so does a value that differs only at ``t1 = t2``
    (:func:`_cut_moves_c_integral`).  So does a C with no feature on the
    diagonal that varies too fast for the probe's nodes:
    ``1 / ((1 + t1)(1 + t2))`` from ``T = 30``, and
    ``0.1 e^{-(t1+t2)} cosh((t1-t2)/2) + 0.01`` from ``T = 155`` to ``579``.
    There the split costs accuracy at a fixed node count.

    The jump test follows :func:`_probe_diagonal_cusp`, which reads the same
    jump off κ² from two steps, but ranges over the steps
    :data:`_C_KINK_PROBE_STEPS`, because a C has no known time scale.  For
    each component and each stencil of :data:`_C_KINK_STENCILS`,
    ``|Δ_p(h)| / h^p`` is level in ``h`` on a jump and falls as ``h`` on a
    smooth function.  An entry has a jump at a time when two consecutive
    ratios are level and no finer rung falls again.  A rung whose
    difference is within rounding of its terms is
    skipped.  One entry must show the jump at two of the times: a C
    computed with noise above rounding gives ratios that are level by
    chance.

    Components are compared one by one, so a weak cusp in one entry is not
    judged against the scale of another.  A C that raises, or returns a
    value that is not finite, is treated as kinked (the safe side, as in
    :func:`_probe_diagonal_cusp`): at the same nodes a split on a C with no
    kink loses little, and one that is missed can lose a lot.

    Limits: a cusp below about 1e-6 of the size of C, relative to C's
    curvature over the range, is not seen.  A pulse of noise narrower than
    about 1e-3 of the range is not seen when it lies between the times, and
    one narrower than about 1e-4 is not seen at all.  A feature that vanishes
    on the diagonal itself is not seen below about 1e-4 of the range.
    """
    t_lo, t_hi = float(times[0]), float(times[1])
    span = t_hi - t_lo
    if not span > 0.0:
        return False
    steps = span * np.asarray(_C_KINK_PROBE_STEPS)
    offsets = np.arange(-2, 3, dtype=float)
    centres = t_lo + span * np.asarray(_C_KINK_PROBE_TIMES)
    t1 = np.repeat(centres, steps.size * offsets.size)
    t2 = t1 + np.tile(np.outer(steps, offsets).ravel(), centres.size)
    try:
        vals = _closed_form_samples(cache, t1, t2, x1, x2)
    except Exception:  # noqa: BLE001 (a user callable; see the docstring)
        return True
    if not np.all(np.isfinite(vals)):
        return True
    vals = vals.reshape(centres.size, steps.size, offsets.size, -1)
    for order, weights in _C_KINK_STENCILS:
        w = np.asarray(weights)
        diff = np.abs(np.einsum("k,abkc->abc", w, vals))
        terms = np.einsum("k,abkc->abc", np.abs(w), np.abs(vals))
        usable = diff > 8.0 * np.finfo(float).eps * terms
        scaled = diff / (steps**order)[None, :, None]
        hits = _level_at_finest_rungs(scaled, usable)
        if np.any(np.sum(hits, axis=0) >= 2):
            return True
    try:
        return _cut_moves_c_integral(cache, t_lo, t_hi, centres, x1, x2)
    except Exception:  # noqa: BLE001 (a user callable; see the docstring)
        return True


def _cut_moves_c_integral(
    cache: Any, t_lo: float, t_hi: float, centres: np.ndarray, x1: Any, x2: Any
) -> bool:
    """Whether a cut at ``s = t`` is worth having in ``∫ C(t, s) ds`` over the
    range, in any entry of C at any time ``t`` of ``centres``.

    The integrators ask this of a many-dimensional integral; here it is put
    to one line through the diagonal.  ``_C_CUT_PROBE_NODES`` nodes on the
    whole range are compared with as many on each side of the cut.  The count
    is twice what an integrator typically spends per piece, so a C that varies
    fast over the range is resolved both ways until it outruns those nodes
    (:func:`_probe_c_diagonal_kink` gives the limits), while a feature on the
    diagonal is resolved only by the cut.  A value that is not finite counts
    as a yes.

    Each time is moved to the nearest node of the whole-range rule.  An
    unsplit tensor rule puts the same nodes on both axes, so some of them lie
    exactly on ``t1 = t2`` and read a feature there at full height, however
    narrow it is, with an error of order ``1 / n`` where the feature has no
    weight in the integral.  The rule of the whole range here does the same,
    and the cut one has no node on the diagonal.
    """
    from numpy.polynomial.legendre import leggauss

    nodes, weights = leggauss(_C_CUT_PROBE_NODES)
    grid = 0.5 * (t_hi + t_lo) + 0.5 * (t_hi - t_lo) * nodes
    centres = grid[np.abs(grid[:, None] - centres).argmin(axis=0)]
    pieces = [(np.full(centres.shape, t_lo), np.full(centres.shape, t_hi))]
    pieces += [(np.full(centres.shape, t_lo), centres)]
    pieces += [(centres, np.full(centres.shape, t_hi))]
    halves = [0.5 * (hi - lo) for lo, hi in pieces]
    s = np.stack(
        [
            0.5 * (hi + lo)[:, None] + half[:, None] * nodes
            for (lo, hi), half in zip(pieces, halves)
        ],
        axis=1,
    )
    t = np.broadcast_to(centres[:, None, None], s.shape)
    vals = _closed_form_samples(cache, t.ravel(), s.ravel(), x1, x2)
    if not np.all(np.isfinite(vals)):
        return True
    vals = vals.reshape(s.shape + (-1,))
    w = np.stack(halves, axis=1)[:, :, None] * weights
    signed = np.einsum("abk,abkc->abc", w, vals)
    sizes = np.einsum("abk,abkc->abc", w, np.abs(vals))
    whole, cut = signed[:, 0], signed[:, 1] + signed[:, 2]
    floor = 8.0 * np.finfo(float).eps * (sizes[:, 0] + sizes[:, 1] + sizes[:, 2])
    return bool(np.any(np.abs(whole - cut) > _C_CUT_PROBE_TOL * np.abs(cut) + floor))


def _level_at_finest_rungs(scaled: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """For ``scaled`` of shape ``(times, rungs, entries)``, coarse to fine:
    whether two consecutive rung ratios are level with no fall after them.
    Returns ``(times, entries)``."""
    both = usable[:, :-1] & usable[:, 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(both, scaled[:, 1:] / scaled[:, :-1], np.nan)
    lo, hi = _C_KINK_LEVEL_BAND
    level = (ratio > lo) & (ratio < hi)
    falling = ratio <= lo
    # falling_after[:, i] : a ratio finer than rung pair i falls.
    falling_after = np.logical_or.accumulate(falling[:, ::-1], axis=1)[:, ::-1]
    runs = level[:, :-1] & level[:, 1:]
    runs[:, :-1] &= ~falling_after[:, 2:]
    return np.any(runs, axis=1)


def _closed_form_samples(
    cache: Any, t1: np.ndarray, t2: np.ndarray, x1: Any, x2: Any
) -> np.ndarray:
    """``(n, ...)`` values the integrators would read from a closed-form C
    at the times ``(t1[i], t2[i])`` and fixed positions, through whichever
    contract the callable was given (see
    ``PropagatorCache._closed_form_at_batch_diag``)."""
    n = t1.size

    def batched(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        return np.tile(arr, (n,) + (1,) * arr.ndim)

    if getattr(cache, "_closed_form_vectorized", False):
        full = np.asarray(
            _C_value_direct_on_copies(cache, batched(x1), t1, batched(x2), t2),
            dtype=float,
        )
    else:
        full = np.stack(
            [
                np.asarray(
                    _C_value_direct_on_copies(cache, x1, float(a), x2, float(b)),
                    dtype=float,
                )
                for a, b in zip(t1, t2)
            ]
        )
    if getattr(cache.model, "diag_C", False) and full.ndim == 3:
        return np.einsum("iaa->ia", full)
    return full


def _coupling_kink_candidates(
    spatial: "SpatialStructure",
    dynamic_coupling: Any,
) -> tuple[tuple[str, str], ...]:
    """Pairs of time variables at which a callable coupling is kinked.

    A coupling callable that sets ``has_coincident_time_kinks = True``
    declares that it is kinked wherever two of its time arguments
    coincide, as a ``min`` over leg times makes it (a raw cumulant of
    exponential pulses) or a ``min`` over partner times (an R-contracted
    cumulant).  Its time arguments are the legs of each of its occurrences
    (``DynamicCouplingPromise.spatial_args_by_name``) resolved through
    ``equal_time_aliases``, the way the integrators resolve them: a raw
    vertex's own leg times, the partner times of an
    ``already_R_contracted`` vertex, and the one shared time of an
    ``equal_time`` vertex, which gives no pair.  :func:`_kink_pairs` keeps
    the pairs of distinct integration variables that are still unordered.
    """
    if dynamic_coupling is None:
        return ()
    from itertools import combinations

    alias = dict(spatial.equal_time_aliases or ())
    out: list[tuple[str, str]] = []
    for key, fn in dynamic_coupling.dynamic_values.items():
        if not _declares(fn, "has_coincident_time_kinks"):
            continue
        legs = dynamic_coupling.spatial_args_by_name[key]
        out.extend(combinations([alias.get(leg, leg) for leg in legs], 2))
    return tuple(out)


def _later_sets(ivars: set, orderings) -> dict[str, set]:
    """``{v: every variable forced later than v}``: the transitive closure
    of the ``(earlier, later)`` edges among ``ivars``."""
    closed = _causal_reachability(
        tuple((e, l) for e, l in orderings if e in ivars and l in ivars)
    )
    return {v: closed.get(v, set()) for v in ivars}


def _kink_candidates(
    spatial: "SpatialStructure",
    c_kink: bool,
    orderings,
    coupling_pairs=(),
    keep=None,
    swept=(),
) -> list[tuple[str, str]]:
    """Pairs of time labels whose crossing kinks the integrand:

    * two parents of one integration variable: its upper bound
      ``min(parents)`` changes branch where they cross.  A vertex with
      several ψ legs at one time (an ``equal_time`` non-local vertex or a
      local vertex with two ψ legs) has such parents, and so can a
      variable that an earlier split placed below two others;
    * with ``c_kink``, the two ends of a C propagator, when C is kinked on
      its time diagonal (see :func:`_c_has_diagonal_kink`);
    * ``coupling_pairs``: two time arguments of a coupling callable that
      declares ``has_coincident_time_kinks`` (see
      :func:`_coupling_kink_candidates`).

    ``keep`` selects which labels count as a parent: :func:`_kink_pairs`
    passes the integration variables, :func:`_kink_constant_cuts` passes
    those and the externals pinned at a fixed time.  Both callers drop
    afterwards the pairs they cannot act on.
    """
    from itertools import combinations

    alias = dict(spatial.equal_time_aliases or ())
    ivars = set(spatial.time_integration_vars)
    keep = ivars.__contains__ if keep is None else keep
    parents: dict[str, set] = defaultdict(set)
    for earlier, lat in orderings:
        if earlier in ivars and keep(lat):
            parents[earlier].add(lat)
    candidates = [
        pair for ps in parents.values() for pair in combinations(sorted(ps), 2)
    ]
    if swept:
        # Internal lower limits and earliest-first swept external ranges
        # contain max(external predecessors), just as upper limits contain
        # min(parents). Resolve the same full closure as the bound mapping;
        # an earlier external can reach the variable through an internal.
        reachable = _causal_reachability(tuple(orderings))
        for var in ivars | set(swept):
            sources = [
                v
                for v in spatial.external_points
                if keep(v) and var in reachable.get(v, ())
            ]
            candidates.extend(combinations(sorted(sources), 2))
    if c_kink:
        candidates += [
            (alias.get(sp_l, sp_l), alias.get(sp_r, sp_r))
            for sp_l, sp_r, _il, _ir in spatial.c_propagators
        ]
    candidates += list(coupling_pairs)
    return candidates


def _split_vars(spatial: "SpatialStructure", swept=()) -> set:
    """The times an extra ordering can be placed between: the integration
    variables, plus the externals ``integrate_over`` sweeps.

    A swept external is drawn before every internal variable, so an
    ordering with one is carried by the mapping already, as an upper
    bound (``min(parents)``, the swept column) or a lower one (a
    :func:`_causal_lower_bound_sources` source).  Two swept externals are
    a pair as well: :func:`_swept_external_order` draws them, and the
    integrators that split pass it the orderings the split added, so an
    extra ordering reaches the sampler.
    """
    return set(spatial.time_integration_vars) | set(swept)


def _kink_pairs(
    spatial: "SpatialStructure",
    c_kink: bool,
    orderings=None,
    coupling_pairs=(),
    swept=(),
) -> list[tuple[str, str]]:
    """Times that ``orderings`` (default: the causal structure) leave
    unordered and whose crossing kinks the integrand inside the domain (the
    three sources are listed in :func:`_kink_candidates`).

    Both ends must be in :func:`_split_vars`.  A pair of two swept
    externals is kept when :func:`_swept_external_order` leaves their
    relative order open: only that function orders two swept externals in
    the samplers, and the integrators that split pass it the orderings the
    split added, so an extra ordering reaches it.  A kink between an
    integration variable and an external point pinned at a fixed time is a
    constant cut rather than an ordering; it is
    :func:`_kink_constant_cuts`.
    """
    orderings = spatial.time_orderings if orderings is None else orderings
    swept = set(swept)
    nodes = _split_vars(spatial, swept)
    later = _later_sets(nodes, orderings)
    candidates = _kink_candidates(
        spatial, c_kink, orderings, coupling_pairs, keep=nodes.__contains__, swept=swept
    )
    full_later: dict[str, set] | None = None
    pairs: list[tuple[str, str]] = []
    for a, b in candidates:
        if a == b or a not in nodes or b not in nodes:
            continue
        if a in swept and b in swept:
            # Two swept externals are drawn by _swept_external_order,
            # which reads the FULL causal closure: unlike _later_sets it
            # follows a chain through a point that is not itself a split
            # variable (a fixed external, say), and such a chain orders
            # the two just as firmly.  Split only a pair it leaves open.
            if full_later is None:
                full_later = _causal_reachability(tuple(orderings))
            if b in full_later.get(a, ()) or a in full_later.get(b, ()):
                continue
        elif b in later[a] or a in later[b]:
            continue
        pair = tuple(sorted((a, b)))
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def _kink_orientations(
    spatial: "SpatialStructure", pairs: list[tuple[str, str]], orderings=None, swept=()
) -> list[tuple]:
    """Every order of the kink pairs consistent with ``orderings`` (default:
    the causal structure), each as a tuple of extra ``(earlier, later)``
    edges.  The sub-domains they define partition the domain.

    An orientation is dropped when it makes the causal closure cyclic.  The
    closure through the split variables is checked first; for a swept
    external a chain through a point that is not itself a split variable (a
    fixed external, say) also counts, because :func:`_swept_external_order`
    follows that chain when it draws the swept externals, and it refuses a
    loop (:func:`_refuse_a_time_loop`).
    """
    from itertools import product

    orderings = spatial.time_orderings if orderings is None else orderings
    nodes = _split_vars(spatial, swept)
    base = [(e, l) for e, l in orderings if e in nodes and l in nodes]
    out = []
    for bits in product((0, 1), repeat=len(pairs)):
        extra = tuple(
            (a, b) if bit == 0 else (b, a) for (a, b), bit in zip(pairs, bits)
        )
        later = _later_sets(nodes, base + list(extra))
        if any(v in later[v] for v in nodes):  # cycle among the nodes
            continue
        if swept:
            full = _causal_reachability(tuple(orderings) + extra)
            if any(v in full.get(v, ()) for v in swept):
                continue
        out.append(extra)
    return out


def _cut_bounds(extra_bounds) -> tuple[dict, dict]:
    """``({var: upper}, {var: lower})`` of the cuts made so far, each cut a
    ``(variable, "le" | "ge", time)`` triple."""
    upper: dict[str, float] = {}
    lower: dict[str, float] = {}
    for var, side, t_star in extra_bounds:
        target = upper if side == "le" else lower
        pick = min if side == "le" else max
        t_star = float(t_star)
        target[var] = t_star if var not in target else pick(target[var], t_star)
    return upper, lower


def _applied_cut_bounds(
    spatial: "SpatialStructure", orderings, extra_bounds, swept=()
) -> tuple[dict, dict]:
    """The cut constants the integrators fold into each variable's range,
    carried to the variables an ordering puts on the same side of the cut.

    ``v ≥ t*`` and ``v ≤ w`` give ``w ≥ t*``.  The causal mapping does not
    carry that on its own: an upper bound is ``min(parents)``, which carries
    a cut downwards, while a lower bound is per variable.  Left out, the
    piece above ``t*`` gives its outer variable a range that collapses to
    zero width at ``t*``: the kink is moved, not removed (measured on
    demo 6's ``F F`` channel at distinct external times: 5.5e-05 at 16
    nodes, machine precision with the bound carried).  ``v ≤ t*`` carries
    upwards for symmetry, where ``min(parents)`` already holds it.

    Empty in, empty out, so a diagram with no cut takes the mapping it
    always took.
    """
    upper, lower = _cut_bounds(extra_bounds)
    if not upper and not lower:
        return upper, lower
    ivars = _split_vars(spatial, swept)
    later = _later_sets(ivars, orderings)
    out_up, out_lo = dict(upper), dict(lower)
    for v, t_star in lower.items():
        for w in later.get(v, ()):  # v ≤ w, so w ≥ t*
            out_lo[w] = max(out_lo[w], t_star) if w in out_lo else t_star
    for v, t_star in upper.items():
        for u in ivars:
            if v in later.get(u, ()):  # u ≤ v, so u ≤ t*
                out_up[u] = min(out_up[u], t_star) if u in out_up else t_star
    return out_up, out_lo


def _constant_bounds(
    spatial: "SpatialStructure",
    orderings,
    fixed_times,
    t_min: float,
    t_ceiling: float,
    extra_bounds=(),
    swept=(),
    swept_ceiling=None,
) -> tuple[dict, dict]:
    """``({var: upper}, {var: lower})``: the constants each integration
    variable is already held between, under ``orderings`` and the cuts made
    so far.

    An external pinned at its own time bounds every variable ordered
    against it, and the bound carries along the causal edges: ``u ≤ v`` and
    ``v ≤ t_y`` bound ``u`` by ``t_y`` too.  A variable whose bounds already
    put it on one side of a kink's time needs no cut there.  A lower bound
    that only a *swept* external carries is not a constant and is left out,
    which can only ask for a cut that turns out to be empty.
    """
    ivars = _split_vars(spatial, swept)
    upper = {
        v: float(
            swept_ceiling if v in swept and swept_ceiling is not None else t_ceiling
        )
        for v in ivars
    }
    lower = {v: float(t_min) for v in ivars}
    for earlier, lat in orderings:
        if earlier in ivars and lat in fixed_times:
            upper[earlier] = min(upper[earlier], float(fixed_times[lat]))
        if lat in ivars and earlier in fixed_times:
            lower[lat] = max(lower[lat], float(fixed_times[earlier]))
    cut_up, cut_lo = _cut_bounds(extra_bounds)
    for var, t_star in cut_up.items():
        if var in ivars:
            upper[var] = min(upper[var], t_star)
    for var, t_star in cut_lo.items():
        if var in ivars:
            lower[var] = max(lower[var], t_star)
    later = _later_sets(ivars, orderings)
    up = {v: min([upper[v]] + [upper[w] for w in later[v]]) for v in ivars}
    lo = {
        v: max([lower[v]] + [lower[w] for w in ivars if v in later[w]]) for v in ivars
    }
    return up, lo


def _kink_constant_cuts(
    spatial: "SpatialStructure",
    c_kink: bool,
    bounds: tuple[dict, dict],
    fixed_times,
    orderings=None,
    coupling_pairs=(),
    extra_bounds=(),
    swept=(),
) -> list[tuple[str, float]]:
    """``(variable, time)`` cuts: an integration variable whose range
    crosses a time at which the integrand, or the variable's own upper
    bound, is kinked.

    Sources:

    * the kinks of :func:`_kink_candidates` with one end an external in
      ``fixed_times`` instead of a second integration variable;
    * a cut already made on a variable ``v`` (``extra_bounds``, ``v ≤ t*``)
      is one more term in ``v``'s upper bound ``min(parents, t*)``, which
      changes branch where a *variable* parent crosses ``t*``, the same
      kink the two-parent rule of :func:`_kink_candidates` covers, with a
      constant for one of the parents.  Left out, cutting the inner
      variable only moves the kink onto the outer one.
    * a lower cut leaves ``max(swept_predecessor, t*)`` in a later
      variable's lower limit; that earlier swept external must also be cut.

    A constant is not an ordering between two variables, so the cut cannot
    be expressed as an extra edge: the two pieces are ``[t*, min(parents)]``
    and ``[t_min, min(t*, parents)]``, carried by the integrators'
    ``_extra_bounds``.  ``bounds`` is :func:`_constant_bounds`; a variable
    already held on one side of ``t*`` is skipped, which keeps a diagram
    whose kink lies on the boundary of its domain (every external at one
    time, the usual case) unsplit and bit-identical.
    """
    ivars = _split_vars(spatial, swept)
    upper, lower = bounds
    orderings = spatial.time_orderings if orderings is None else orderings
    candidates = _kink_candidates(
        spatial,
        c_kink,
        orderings,
        coupling_pairs,
        keep=lambda label: label in ivars or label in fixed_times,
        swept=swept,
    )
    cut_times: list[tuple[str, float]] = []
    for a, b in candidates:
        for var, other in ((a, b), (b, a)):
            if var not in ivars or other in ivars or other not in fixed_times:
                continue
            cut_times.append((var, float(fixed_times[other])))
    cut_upper, cut_lower = _cut_bounds(extra_bounds)
    for v, t_star in cut_upper.items():
        for earlier, lat in orderings:
            if earlier == v and lat in ivars:
                cut_times.append((lat, t_star))
    # Swept externals are drawn earliest-first, unlike internal variables.
    # A lower cut on a later variable leaves max(earlier_swept, t*) in its
    # range, kinked where that earlier sample crosses t*. The source can be
    # a transitive predecessor, as in an external-response/internal chain.
    if swept and cut_lower:
        reachable = _causal_reachability(tuple(orderings))
        # The applied bounds, so a variable that holds the cut only through
        # an ordering counts: ``v ≥ t*`` and ``v ≤ w`` with a swept ``s ≤ w``
        # leave ``max(s, t*)`` in ``w``'s range.
        _, applied_lower = _applied_cut_bounds(
            spatial, orderings, extra_bounds, swept=swept
        )
        for v, t_star in applied_lower.items():
            for source in swept:
                if v in reachable.get(source, ()):
                    cut_times.append((source, t_star))
    cuts: list[tuple[str, float]] = []
    for var, t_star in cut_times:
        if not lower[var] < t_star < upper[var]:
            continue
        if (var, t_star) not in cuts:
            cuts.append((var, t_star))
    return cuts


def _refuse_a_time_loop(time_orderings: list[tuple[str, str]]) -> None:
    """Raise ``ValueError`` when ``(earlier, later)`` edges close a loop.

    The Wick engines drop a loop of R, so this is reached by a hand-built
    :class:`~sft_wick.perturbation.DiagramTerm`, or by an L0 equal-time vertex
    with a phi leg, whose legs close a loop in time and not in labels.  No
    time ordering satisfies such a term and the integrators have no domain
    for it.  An equal-point R never reaches this: it has no edge here and
    gives 0 through Theta(0)=0.
    """
    if not r_vanishes((later, earlier) for earlier, later in time_orderings):
        return
    after = _causal_reachability(time_orderings)
    points = ", ".join(sorted(v for v, later in after.items() if v in later))
    raise ValueError(
        f"The R propagators of this diagram close a time loop through "
        f"{points}: no causal time ordering satisfies them.  Legs of an "
        f"equal-time vertex count as one time.  Leave the term out."
    )


def _alias_representatives(aliases: Any) -> dict[str, str]:
    """``{leg: the end of its chain of aliases}``.

    A leg of an equal-time vertex can be the partner of an absorbed leg, so an
    alias can name another alias.  Everything that reads an alias takes one
    step, and this map is what makes one step enough.
    """
    step = dict(aliases)

    def end(label: str) -> str:
        seen = {label}
        while step.get(label, label) not in seen:
            label = step[label]
            seen.add(label)
        return label

    return {leg: end(leg) for leg in step}


def analyze_spatial(dt: "DiagramTerm") -> SpatialStructure:
    """Analyze a DiagramTerm's propagator topology.

    Determines direction identification groups from R-propagator connectivity,
    time ordering constraints from R causality, and which integration variables
    survive after δ-function elimination.

    Raises:
        ValueError: The R propagators close a loop in time, naming its points.
    """
    integration_set = set(dt.integration_vars)
    # Equal-time non-local vertices share a single time integration
    # variable across their m legs. Build the alias map (leg → canonical
    # representative) and exclude the non-representatives from the
    # surviving time-integration set so the Jacobian is one factor of
    # ``width`` per vertex, not ``width^m``. Spatial labels remain
    # independent (each leg keeps its own direction / position).
    equal_time_aliases = _alias_representatives(
        getattr(dt, "equal_time_aliases", ()) or ()
    )
    time_integration_set = set(integration_set)
    for non_rep in equal_time_aliases:
        time_integration_set.discard(non_rep)

    # Classify propagators
    r_props: list[tuple[str, str]] = []
    c_props: list[tuple[str, str, str | None, str | None]] = []
    for p in dt.propagators:
        if p.kind == "R":
            r_props.append((p.spatial_left, p.spatial_right))
        else:
            c_props.append(
                (p.spatial_left, p.spatial_right, p.index_left, p.index_right)
            )

    # --- Direction groups via union-find on R propagators ---
    all_points: set[str] = set()
    for p in dt.propagators:
        all_points.add(p.spatial_left)
        all_points.add(p.spatial_right)

    parent: dict[str, str] = {p: p for p in all_points}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Prefer external point as root
            if rb not in integration_set and ra in integration_set:
                parent[ra] = rb
            else:
                parent[rb] = ra

    for sl, sr in r_props:
        union(sl, sr)

    # Build groups
    groups_dict: dict[str, set[str]] = defaultdict(set)
    for p in sorted(all_points):
        groups_dict[find(p)].add(p)

    direction_groups = tuple(frozenset(g) for g in groups_dict.values())

    # Direction map: each point → n_{representative}
    direction_map: dict[str, str] = {}
    direction_integration_vars: list[str] = []
    for group in direction_groups:
        # Pick representative: prefer external point, then lexicographic
        external = sorted(p for p in group if p not in integration_set)
        if external:
            rep = external[0]
        else:
            rep = sorted(group)[0]
            direction_integration_vars.append(rep)
        dir_name = f"n_{rep}"
        for p in group:
            direction_map[p] = dir_name

    # --- Time orderings from R causality ---
    # R(a, b): Θ(t_a − t_b) → t_b ≤ t_a → (earlier=b, later=a)
    time_orderings: list[tuple[str, str]] = []
    for sl, sr in r_props:
        time_orderings.append((sr, sl))  # (earlier, later)

    # --- Topological sort of integration time variables ---
    # Only the canonical equal-time representatives (and any non-equal-time
    # internals) appear in time_integration_set; the topo sort drops the
    # filtered non-representatives implicitly because they're absent from
    # the input list and from any (earlier, later) entry whose endpoint is
    # filtered (rewritten below). Time-orderings that touch a filtered
    # label are remapped to its canonical representative so causality
    # between an equal-time vertex and the rest of the diagram remains
    # captured by R-propagators on the surviving (representative) legs.
    time_orderings_collapsed: list[tuple[str, str]] = []
    for earlier, later in time_orderings:
        earlier_alias = equal_time_aliases.get(earlier, earlier)
        later_alias = equal_time_aliases.get(later, later)
        if earlier_alias == later_alias:
            # Same physical time after collapse: no ordering needed.
            continue
        time_orderings_collapsed.append((earlier_alias, later_alias))
    _refuse_a_time_loop(time_orderings_collapsed)
    sorted_time_vars = _topological_sort_times(
        tuple(sorted(time_integration_set)), time_orderings_collapsed
    )

    # --- External points ---
    external_points = tuple(sorted(p for p in all_points if p not in integration_set))

    return SpatialStructure(
        direction_groups=direction_groups,
        direction_map=direction_map,
        time_orderings=tuple(time_orderings_collapsed),
        r_propagators=tuple(r_props),
        c_propagators=tuple(c_props),
        time_integration_vars=tuple(sorted_time_vars),
        direction_integration_vars=tuple(sorted(direction_integration_vars)),
        external_points=external_points,
        equal_time_aliases=tuple(sorted(equal_time_aliases.items())),
        r_absorbed_pairs=tuple(getattr(dt, "r_absorbed_pairs", ()) or ()),
    )


def _conflicting_group_positions(
    spatial: "SpatialStructure",
    positions: Any,
) -> tuple[str, str] | None:
    """The first pair of points of ONE direction group that ``positions``
    places at different coordinates, or ``None`` when every group has a
    single position.

    A direction group is a set of points that R-propagators identify.  R
    carries ``δ(n − n')``, so all of them sit at the same spatial point
    and the group has one coordinate.  Candidates are compared in sorted
    order, so the pair the caller is told about does not itself depend on
    set iteration order.
    """
    if not positions:
        return None
    for group in spatial.direction_groups:
        named = [p for p in sorted(group) if p in positions]
        if len(named) < 2:
            continue
        first = np.asarray(positions[named[0]])
        for p in named[1:]:
            if not np.array_equal(np.asarray(positions[p]), first):
                return (named[0], p)
    return None


def _external_positions(spatial: "SpatialStructure", positions: Any) -> dict:
    """The entries of ``positions`` that name external points of the
    diagram.

    ``positions`` places external points.  A vertex point takes the
    coordinate of its direction group, the points that response
    propagators join, and that coordinate is the group's external
    point's.  A key that names a vertex point used to be read too: it
    replaced the group's coordinate when the external point had no entry,
    and at a different value it raised the conflicting-group error about a
    point the caller had not declared.  Since an observable may name its
    points like the vertex points (``y_0``), one positions dict serving
    several observables can hold such a key, and it is now ignored, like a
    key that names no point of the diagram.
    """
    if not positions:
        return {}
    return {p: positions[p] for p in spatial.external_points if p in positions}


def _require_one_position_per_group(
    spatial: "SpatialStructure",
    positions: Any,
    *,
    where: str,
) -> None:
    """Refuse two points of one direction group placed at two positions.

    The group has one coordinate, and the resolver would take whichever
    of the two it met first in a ``frozenset``, an order that changes
    with the labels and with ``PYTHONHASHSEED``, so the same physics
    spelled with different external names returns different numbers.

    Two *external* points share a group only when the observable carries
    a response leg (a ψ operator) or the action has a local vertex with
    two or more ψ legs (what
    :class:`~sft_wick.workflow.specs.MultiplicativeImpulse` lowers to);
    with one ψ leg per vertex no R chain can join two φ externals.
    """
    pair = _conflicting_group_positions(spatial, positions)
    if pair is None:
        return
    p, q = pair
    raise ValueError(
        f"{where}: the spatial points '{p}' and '{q}' are at different "
        f"positions ({positions[p]!r} and {positions[q]!r}), and a chain of "
        f"response propagators joins them in this diagram.  R carries "
        f"delta(n - n'), so the points of an R-connected group are one "
        f"spatial point with one coordinate; at two positions the diagram is "
        f"a delta function rather than a value, and the resolver would use "
        f"whichever of the two positions it met first in a frozenset, an "
        f"order that changes with the labels and with PYTHONHASHSEED.  Put "
        f"both points at one position, positions={{{p!r}: v, {q!r}: v}}; "
        f"distinct labels and distinct external_times are fine.  An R chain "
        f"joins two external points only when the observable carries a "
        f"response leg (a 'psi' operator) or a local vertex has two or more "
        f"psi legs (what MultiplicativeImpulse lowers to)."
    )


# ---------------------------------------------------------------------------
# Step 3: Propagator Model and Cache
# ---------------------------------------------------------------------------


#: Entries kept in ``PropagatorCache``'s ``C_value`` memo before the oldest is
#: evicted.  It used to be unbounded (described in comments as an LRU while
#: being a plain dict), so a long sweep grew it without limit.
_C_CACHE_MAXSIZE = 65536

#: Two times are treated as ON the C-table diagonal when they differ by no
#: more than this fraction of the table's span (or by rounding of the times).
#: C(t1,t2) has a derivative kink of exactly -sigma2(t) on t1 == t2 (the
#: integral's upper limit is min(t1,t2)), so the substitution error from using
#: the diagonal spline is sigma2 times the tolerance, orders below the table's
#: own accuracy.
_DIAG_TOL = 1e-9
#: Units of rounding of a time that two times of one point can differ by.
_DIAG_ROUNDING = 8.0 * np.finfo(float).eps

#: Tolerances of the ``dblquad`` and ``quad`` rule, under any given in
#: ``quad_opts``.  Relative only, as every other rule: scipy's default
#: ``epsabs=1.49e-8`` stops on the first samples of a small C, so C was not
#: linear in the amplitude of κ², off by a factor 3 at amplitude 1e-9.
_QUAD_TOLERANCES = {"epsabs": 0.0, "epsrel": 1.49e-8}

#: Relative slack on the limits of a C table.  A sampled time or a computed
#: separation can land a few ulps past a limit it equals; this is well above
#: that and far below any distance a table resolves.
_TABLE_EDGE_RTOL = 1e-12

_REALITY_TOL = 1e-9
#: Values below this magnitude are treated as an exact zero (a diagram that
#: cancelled), so a denormal-scale imaginary residue never raises.
_REALITY_FLOOR = 1e-290


def _residue_negligible(worst: float, scale: float) -> bool:
    """Whether an imaginary part of size ``worst`` may be dropped from a value
    whose real part has size ``scale``.  The rule of every reality projection,
    scalar or batched.  A NaN in either is never negligible."""
    return (
        worst == 0.0
        or max(worst, scale) <= _REALITY_FLOOR
        or worst <= _REALITY_TOL * scale
    )


def _finite_or_raise(value: float, where: str) -> float:
    """``value``, refused when it is not finite.

    The batched integrators add their samples without the per-sample
    projection of the scalar loop, so a NaN or an infinity in the samples
    arrives here in their total.
    """
    if isfinite(value):
        return value
    raise ValueError(
        f"Diagram integrand{where} evaluated to {value!r}.  A position, a time "
        f"or a propagator of this integral is not finite."
    )


def _real_or_raise(
    value, e_psi: int = 0, *, scale: float = 0.0, where: str = ""
) -> float:
    """Project a diagram value onto the reals **without guessing a sign**.

    By the MSR reality theorem (see
    :meth:`~sft_wick.perturbation.DiagramTerm.observable_phase_factor`) a
    diagram equals ``i**(-E_psi)`` times a real number, provided each vertex
    coupling carries the ``(+/-i)**n_psi`` factor demanded by
    ``<phi psi> = -i R``.  Rotating by ``i**E_psi`` therefore lands exactly on
    the real axis.

    Anything left over is a mis-specified action, most often a missing MSR
    factor on a multi-psi vertex, for which the required coefficient is
    ``-(i**m)/m!`` (see ``NonLocalVertex.msr_factor``).  That is reported, not
    silently turned into ``abs()`` (which flips the sign of negative
    contributions) or into ``0`` (which reads as "no contribution").

    Args:
        value: raw complex diagram value.
        e_psi: number of external response legs of the observable.
        scale: magnitude of the largest summand contributing to ``value``, so
            the test stays well posed when the diagram cancels to near zero.
        where: short context string included in the error message.
    """
    z = complex(value) * (1j ** int(e_psi))
    re_, im_ = z.real, z.imag
    if _residue_negligible(abs(im_), max(abs(re_), abs(scale))):
        return re_
    raise ValueError(
        f"Diagram integrand{where} evaluated to {value!r}; after the "
        f"i**E_psi rotation (E_psi={e_psi}) it is {z!r}, whose imaginary part "
        f"is not negligible at {_REALITY_TOL:g} relative tolerance. Each "
        f"vertex coupling must carry the (+/-i)**n_psi factor demanded by "
        f"<phi psi> = -i R; for an m-leg all-psi vertex that factor is "
        f"-(i**m)/m! (see NonLocalVertex.msr_factor). Passing a real coupling "
        f"where an imaginary one is required is the usual cause."
    )


@dataclass(frozen=True)
class PropagatorModel:
    """Physical propagator functions for numerical evaluation.

    The R propagator has the form::

        R_{ab}(n,t; n',t') = δ(n−n') Θ(t−t') R_time(t,t') δ_{ab}   [iso_R]
        R_{ab}(n,t; n',t') = δ(n−n') Θ(t−t') R_time_ab(t,t')        [general]

    The C propagator is computed from R and the 2-point cumulant κ^(2)::

        C_{ab}(n₁,t₁; n₂,t₂) = ∫∫ R_time(t₁,λ') κ_{ab}(n₁,λ'; n₂,λ'')
                                      R_time(t₂,λ'') dλ' dλ''

    Attributes:
        R_time: Callable ``(t_left, t_right) → scalar`` when ``iso_R=True``,
            or ``(t_left, t_right) → (N, N) array`` otherwise.  It need not
            vanish for ``t_left < t_right``: the integrators apply Θ, and C
            is 0 when a time is at or below ``t_min``.
        kappa2: Callable ``(n1, t1, n2, t2) → (N, N) array``.
            The 2-point cumulant.
        n_components: Number of field components.
        iso_R: Whether R is isotropic (proportional to identity).
        diag_C: Whether C is diagonal in component indices.
        t_min: Lower bound for the λ integrals in C computation.
    """

    R_time: Callable
    kappa2: Callable
    n_components: int
    iso_R: bool = True
    diag_C: bool = False
    t_min: float = 0.0
    # Optional white-noise component of the source-field correlation:
    #
    #   κ²(t1, x1; t2, x2) = kappa2_smooth + δ(t1 − t2) · sigma2(x1, t1, x2).
    #
    # ``sigma2`` is called as ``sigma2(n1, t, n2)``, with one time since
    # the δ collapses both, and returns an (N, N) matrix.  When provided,
    # ``C`` picks up an additional 1-D integral
    #
    #   C_white(t1, t2; x1, x2) =
    #       ∫_{t_min}^{min(t1,t2)} R(t1, τ) σ²(x1, τ, x2) R(t2, τ) dτ,
    #
    # evaluated by :meth:`PropagatorCache._C_value_direct` and the
    # :meth:`~PropagatorCache.C_value` dblquad fallback alongside the
    # existing 2-D ``kappa2_smooth`` integral.  All
    # ``precompute_C_table_*`` builders call ``_C_value_direct``, so
    # the white-noise term is automatically absorbed into every
    # homogeneity-mode spline with no caller-side changes.
    sigma2: Callable | None = None


def _polynomial_weights(x: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Lagrange weights on a short, dimensionless stencil."""
    weights = np.ones((len(x), len(nodes)))
    for k, node in enumerate(nodes):
        for l, other in enumerate(nodes):
            if k != l:
                weights[:, k] *= (x - other) / (node - other)
    return weights


class _MinLagGrid:
    """Physical samples and an auxiliary extension of a min-time/lag table.

    Only the triangle ``s + u <= t_max`` is sampled from the physical
    kernels.  The rectangular spline needs values beyond that triangle;
    these are polynomial continuations of *table values*, not evaluations
    of R, noise or a user C outside its declared domain.

    For each lag, continue the last four in-domain min-time samples.  The
    two shortest nonzero intervals get three extra samples in total, all
    inside the domain, so their continuation remains cubic.  At maximum
    lag the interval has zero length; continue the preceding lag columns
    instead, preserving its single physical node.  Every original physical
    grid node is retained exactly.  Small full grids use the available
    lower-degree stencil at the last column.
    """

    def __init__(self, ts: np.ndarray):
        self.n = n = len(ts)
        if n < 2:
            raise ValueError("C tables require at least two time grid nodes")
        span = float(ts[-1] - ts[0])
        self.points: list[tuple[float, float]] = []
        self.columns: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for j in range(n):
            count = n - j
            indices = list(range(count))
            if count == 2:
                indices.extend((1.0 / 3.0, 2.0 / 3.0))
            elif count == 3:
                indices.append(0.5)
            start = len(self.points)
            for i in indices:
                earlier = float(ts[0] + span * i / (n - 1))
                # i+j <= n-1 by construction.  Clamp only roundoff at
                # the physical upper boundary, never an out-of-domain node.
                later = float(min(ts[-1], ts[0] + span * (i + j) / (n - 1)))
                self.points.append((later, earlier))
            regular = np.arange(start, start + count)
            order = np.argsort(indices)[-4:]
            self.columns.append(
                (regular, start + order, np.asarray(indices, dtype=float)[order])
            )

    def complete(self, values: np.ndarray) -> np.ndarray:
        """Expand ``(physical samples, *spatial)`` to ``(s, u, *spatial)``."""
        n = self.n
        grid = np.empty((n, n) + values.shape[1:], dtype=values.dtype)
        for j, (regular, support, coords) in enumerate(self.columns):
            count = len(regular)
            grid[:count, j] = values[regular]
            if j < n - 1 and count < n:
                weights = _polynomial_weights(np.arange(count, n), coords)
                grid[count:, j] = np.tensordot(weights, values[support], axes=(1, 0))
        start = max(0, n - 5)
        weights = _polynomial_weights(np.array([n - 1]), np.arange(start, n - 1))[0]
        grid[1:, -1] = np.tensordot(grid[1:, start:-1], weights, axes=(1, 0))
        return grid


class _LazyTimeSplineCache:
    """Per-parameter-value 2-D ``(t1, t2)`` spline cache for lazy
    spatial evaluation.

    Backs the lazy mode of :class:`PropagatorCache` for each
    homogeneity symmetry: instead of pre-building an (n+1)-D spline
    over a pre-allocated parameter grid, this cache builds a 2-D
    time spline on demand for each distinct parameter value seen.

    Parameter semantics per mode:

    - ``translation``: key is ``r = |x1 − x2|`` (scalar ≥ 0).
    - ``rotation``: key is ``cos θ = x1·x2 / (|x1| |x2|)`` (scalar
      in ``[-1, 1]``).
    - ``general``: key is ``(x1_tuple, x2_tuple)``, a pair of
      tuples over all x dimensions.

    Memoization key: :func:`_lazy_key_separation` and
    :func:`_lazy_key_value`, so floating-point near-duplicates collapse and
    a key depends on no units.  Two pairs share a table when their
    separations agree to :data:`LAZY_KEY_DIGITS` digits, so a kernel that
    varies on less than 1e-12 of the separation itself is not resolved.
    """

    def __init__(
        self,
        parent: "PropagatorCache",
        t_max: float,
        n_grid_t: int,
        mode: str,
        n_jobs: int = 1,
    ):
        self.parent = parent
        self.t_max = t_max
        self.n_grid_t = n_grid_t
        self.mode = mode
        self.ts = np.linspace(parent.model.t_min, t_max, n_grid_t)
        #: Lag grid for the ``(min(t1,t2), |t1-t2|)`` reparameterisation.
        #: The kink of C is then the ``u = 0`` axis (a grid boundary), not a
        #: diagonal running through the cells.
        self.us = np.linspace(0.0, t_max - parent.model.t_min, n_grid_t)
        self._splines_by_key: dict = {}
        #: Separable-translation shortcut (see :meth:`_build`): the
        #: ``(L, E)`` grids at zero separation divided by ``κ_x(0)``, built
        #: once and rescaled by ``κ_x(r)`` for every later ``r``.
        self._base_grids: tuple | None = None
        #: Number of times the full quadrature grid was actually built,
        #: exposed so tests and the cost estimator can count table builds.
        self.n_grid_builds: int = 0
        #: Parallel-worker count forwarded to :meth:`_build`.  ``1``
        #: (default) is serial; ``-1`` is all cores via
        #: :mod:`joblib.Parallel` with the ``loky`` backend.  Each
        #: on-demand build for a new parameter value pays a one-time
        #: worker-startup cost (~1 s) before the ``n_grid_t²``
        #: independent ``_C_value_direct`` calls fan out across
        #: cores.
        self.n_jobs = n_jobs

    def get_splines(self, x1_val, x2_val) -> list:
        """Return the list of per-component 2-D splines for this
        (x1, x2) pair.  Builds and memoizes on first call."""
        key = self._make_key(x1_val, x2_val)
        if key not in self._splines_by_key:
            if self.mode == "translation":
                self.parent._require_even_separation([(x1_val, x2_val)])
            self._splines_by_key[key] = self._build(x1_val, x2_val)
        return self._splines_by_key[key]

    def _make_key(self, x1_val, x2_val):
        """Derive the memoization key.  In translation mode the key is the
        separation, in rotation mode the distances of ``x2`` from ``x1`` and
        from its antipode, as unit vectors, which unlike a cosine resolve a
        small angle at both poles, and in general mode both endpoints with
        their separation."""
        if self.mode == "translation":
            # Translation invariance: C depends only on ``||x1 - x2||``.
            # Accept scalar or arbitrary-dimensional vector inputs;
            # the cache shape stays (t1, t2, r) regardless of d.
            return ("r", _lazy_key_separation(x1_val, x2_val))
        if self.mode == "rotation":
            u1, u2 = _unit_vector(x1_val), _unit_vector(x2_val)
            return (
                "angle",
                _lazy_key_separation(u1, u2),
                _lazy_key_separation(u1, -u2),
            )
        # general: keep both endpoints
        a = tuple(_lazy_key_value(v) for v in np.ravel(x1_val).tolist())
        b = tuple(_lazy_key_value(v) for v in np.ravel(x2_val).tolist())
        return ("xx", a, b, _lazy_key_separation(x1_val, x2_val))

    def _build(self, x1_val, x2_val) -> list:
        """Build 2-D ``(t1, t2)`` splines at this specific parameter value.

        Two shortcuts, both exact up to floating-point rounding:

        * **Separability** (translation mode, ``κ²(1, 2) = κ_t · κ_x(r)``,
          no white noise): ``C(r; t1, t2) = κ_x(r) · C(0; t1, t2)``, so the
          quadrature grid is built ONCE at zero separation and rescaled
          for every further ``r``.  The lazy cache used to redo the full
          ``n_grid_t²`` quadrature per distinct ``r``: 4× the work for a
          four-point moment, 12×+ in a positions sweep.
        * **Time symmetry**: the table stores the later-first value
          ``T_ab(s, u) = C_ab(s + u, s)`` with ``s = min(t1, t2)`` and
          ``u = |t1 - t2|``, and the lookup recovers the other order from
          ``C_ab(t1, t2) = C_ba(t2, t1)`` when the kernel declares it
          (:meth:`PropagatorCache._c_time_symmetric`).

        Whether either applies is decided by the parent cache
        (:meth:`PropagatorCache._lazy_spatial_factor` /
        :meth:`PropagatorCache._c_time_symmetric`); a user-supplied
        closed form or C callable gets neither, since nothing is known
        about its structure.

        Parallelises across grid points when :attr:`n_jobs` ≠ 1, with a
        progress bar in either case.
        """
        x1_arr = np.asarray(x1_val)
        x2_arr = np.asarray(x2_val)
        grids = egrids = None
        if self.mode == "translation":
            factor_fn = self.parent._lazy_spatial_factor()
            if factor_fn is not None:
                grids, egrids = self._grids_from_base(x1_arr, x2_arr, factor_fn)
        if grids is None:
            grids, egrids = self._grids_by_quadrature(x1_arr, x2_arr)
        return self._splines_from_grids(grids, egrids)

    def _grids_from_base(self, x1_arr, x2_arr, factor_fn):
        """Rescale the zero-separation grids by ``κ_x(r) / κ_x(0)``.

        Returns ``(None, None)`` (caller falls back to quadrature) when
        ``κ_x(0)`` vanishes, is not finite or raises, so the shortcut can never
        divide by zero.
        """
        f_r = float(factor_fn(_separation(x1_arr, x2_arr)))
        if self._base_grids is None:
            try:
                f0 = float(factor_fn(0.0))
            except Exception:  # noqa: BLE001 (a user callable)
                return None, None
            if not np.isfinite(f0) or f0 == 0.0:
                return None, None
            zero = np.zeros_like(np.asarray(x1_arr, dtype=float))
            base_L, base_E = self._grids_by_quadrature(zero, zero)
            self._base_grids = (
                [g / f0 for g in base_L],
                None if base_E is None else [g / f0 for g in base_E],
            )
        base_L, base_E = self._base_grids
        return (
            [f_r * g for g in base_L],
            None if base_E is None else [f_r * g for g in base_E],
        )

    def _grids_by_quadrature(self, x1_arr, x2_arr):
        """``(later_first, earlier_first)`` grids per stored entry from
        ``_C_value_direct``: ``C_aa`` under ``diag_C``, every ``C_ab``
        otherwise (see :meth:`PropagatorCache._c_table_entries`).

        Grid node ``(i, j)`` is ``L_ab(s_i, u_j) = C_ab(x1, s_i + u_j;
        x2, s_i)`` at ``s_i = ts[i]``, ``u_j = us[j]`` for the
        later-first table, and ``E_ab(s_i, u_j) = C_ab(x1, s_i; x2,
        s_i + u_j)`` for the earlier-first one, built when
        :meth:`PropagatorCache._need_earlier_first` says so; otherwise the caller recovers
        ``E`` from the transposed entry of the later-first table.

        :class:`_MinLagGrid` supplies only in-domain physical samples and
        extends their values for the rectangular spline.  No physical
        kernel is called beyond ``t_max``.
        """
        from .progress import progress_map

        parent = self.parent
        entries = parent._c_table_entries()
        layout = _MinLagGrid(self.ts)
        need_E = self.parent._need_earlier_first(self.mode)

        tasks = []
        for i, (later, earlier) in enumerate(layout.points):
            tasks.append((0, i, later, earlier))
            # At ``later == earlier`` both halves hold the same cell.
            if need_E and later != earlier:
                tasks.append((1, i, earlier, later))

        direct_kwargs = parent._table_kwargs(x1_arr, x2_arr, self.t_max)

        def _point(args):
            kind, i, t1, t2 = args
            C_mat = np.asarray(
                _C_value_direct_on_copies(
                    parent, x1_arr, t1, x2_arr, t2, **direct_kwargs
                )
            )
            return kind, i, np.array([C_mat[a, b] for a, b in entries])

        label = parent._c_source_label()
        results = progress_map(
            _point,
            tasks,
            f"C table ({label})",
            n_jobs=self.n_jobs,
            unit="cell",
        )
        self.n_grid_builds += 1

        grids = [np.empty(len(layout.points)) for _ in entries]
        egrids = [np.empty(len(layout.points)) for _ in entries] if need_E else None
        for kind, i, cvec in results:
            target = grids if kind == 0 else egrids
            for k, _e in enumerate(entries):
                target[k][i] = cvec[k]
        diagonal = [i for i, (later, earlier) in enumerate(layout.points)
                    if later == earlier]
        for e, g in zip(egrids or (), grids):
            e[diagonal] = g[diagonal]
        return (
            [layout.complete(g) for g in grids],
            None if egrids is None else [layout.complete(g) for g in egrids],
        )

    def _splines_from_grids(self, grids: list, egrids: list | None) -> list:
        from scipy.interpolate import RectBivariateSpline

        ts, us = self.ts, self.us
        raw = [RectBivariateSpline(ts, us, g) for g in grids]
        eraw = (
            None if egrids is None else [RectBivariateSpline(ts, us, g) for g in egrids]
        )
        entries = self.parent._c_table_entries()
        where = {e: k for k, e in enumerate(entries)}
        out = []
        for a, b in entries:
            if eraw is None:
                # E_ab = L_ba at a swap-invariant key (covariance).
                out.append(_MinLagSpline(raw[where[(a, b)]], raw[where[(b, a)]]))
            else:
                out.append(_MinLagSpline(raw[where[(a, b)]], eraw[where[(a, b)]]))
        return out


class _DiagLineSpline:
    """Picklable 1-D cubic spline of a C-table diagonal.

    Two things are true at once and neither alone gives a usable answer:

    * ``scipy.interpolate.CubicSpline`` is NOT picklable (scipy 1.18:
      ``TypeError: cannot pickle 'module' object``), so a
      :class:`PropagatorCache` holding one cannot be sent through joblib or
      saved, a regression against ``RectBivariateSpline``, which is.
    * ``RegularGridInterpolator(method='cubic')`` IS picklable but DIVERGES on
      this data as the grid refines: mid-cell relative error 4.0e-04 at
      n_grid=41 but 8.2e-03 at n_grid=321.  Swapping to it trades a
      serialisation bug for a numerical one.

    So keep the CubicSpline and make the *container* picklable: only the nodes
    and values are serialised, and the spline is rebuilt on first use.
    """

    __slots__ = ("_ts", "_v", "_cs")

    def __init__(self, ts, values):
        object.__setattr__(self, "_ts", np.asarray(ts, dtype=float))
        object.__setattr__(self, "_v", np.asarray(values, dtype=float))
        object.__setattr__(self, "_cs", None)

    def _spline(self):
        if self._cs is None:
            from scipy.interpolate import CubicSpline

            object.__setattr__(self, "_cs", CubicSpline(self._ts, self._v))
        return self._cs

    def __call__(self, t):
        return np.asarray(self._spline()(np.asarray(t, dtype=float)), dtype=float)

    def __getstate__(self):
        return (self._ts, self._v)

    def __setstate__(self, state):
        object.__setattr__(self, "_ts", state[0])
        object.__setattr__(self, "_v", state[1])
        object.__setattr__(self, "_cs", None)


def _diag_line_interp(ts, values):
    """Picklable 1-D cubic interpolator for a C-table diagonal."""
    return _DiagLineSpline(ts, values)


def _diag_line_eval(itp, t):
    """Evaluate a :func:`_diag_line_interp` interpolator at scalar or array t."""
    return np.atleast_1d(itp(np.atleast_1d(np.asarray(t, dtype=float))))


class _MinLagTable:
    """The two halves of a ``C_ab`` table on ``(s, u) = (min(t1, t2),
    |t1 - t2|)``, which fold the time orders onto one point."""

    __slots__ = ()

    @staticmethod
    def _fold(t1: Any, t2: Any) -> tuple:
        """``(s, u, later_first)`` of two times."""
        return np.minimum(t1, t2), np.abs(t1 - t2), t1 >= t2

    def __getstate__(self):
        return {n: getattr(self, n) for n in self.__slots__}

    def __setstate__(self, state):
        for n, v in state.items():
            object.__setattr__(self, n, v)


class _MinLagSpline(_MinLagTable):
    """A ``C_ab`` table reparameterised by ``(s, u) = (min(t1,t2), |t1-t2|)``.

    ``C`` has a derivative discontinuity across ``t1 = t2``.  On a
    ``(t1, t2)`` tensor-product grid that kink runs diagonally through the
    cells, which a tensor-product spline cannot represent: the lazy table lost
    four orders of magnitude within one grid spacing of the diagonal (measured
    1.1e-02 at ``n_grid_t = 41``).  In ``(s, u)`` the same kink is the
    boundary ``u = 0`` and the spline is smooth in the interior.

    ``(s, u)`` folds the two time orders onto one point, so ``_ge`` is the
    later-first spline ``L_ab(s, u) = C_ab(x1, s+u; x2, s)`` and ``_lt``
    the earlier-first one ``E_ab(s, u) = C_ab(x1, s; x2, s+u)``.  At a
    swap-invariant spatial key a declared transposition
    ``C_ab(t1, t2) = C_ba(t2, t1)`` makes ``E_ab = L_ba``, so the caller
    passes the transposed entry and only one grid is built.
    """

    __slots__ = ("_ge", "_lt")

    def __init__(self, later_first, earlier_first):
        self._ge = later_first
        self._lt = earlier_first

    def __call__(self, t1, t2, grid=False):
        if grid:
            raise ValueError("_MinLagSpline does not implement grid=True")
        t1a = np.asarray(t1, dtype=float)
        t2a = np.asarray(t2, dtype=float)
        s, u, ge = self._fold(t1a, t2a)
        a = np.asarray(self._ge(s, u, grid=False), dtype=float)
        b = np.asarray(self._lt(s, u, grid=False), dtype=float)
        out = np.where(ge, a, b)
        return float(out) if out.ndim == 0 else out


class _MinLagInterp(_MinLagTable):
    """A full-grid C_ab table in ``(s, u, *spatial)`` coordinates.

    ``_ge`` evaluates ``L_ab(s, u, *spatial) = C_ab(x1, s+u; x2, s)`` and
    ``_lt`` the earlier-first half ``E_ab(s, u, *spatial) = C_ab(x1, s;
    x2, s+u)``.  At a swap-invariant spatial key (translation's r,
    rotation's cos) with a declared symmetry ``E_ab = L_ba``, so ``_lt`` is
    the transposed entry.  ``__call__`` matches
    ``RegularGridInterpolator.__call__``, an ``(n, ndim)`` point array.
    """

    __slots__ = ("_ge", "_lt")

    def __init__(self, later_first, earlier_first):
        self._ge = later_first
        self._lt = earlier_first

    @staticmethod
    def _evaluate(interp, pts):
        """Linear interpolation uses only physical time-cell vertices.

        In off-diagonal (t1,t2) cells this is the usual bilinear rule.
        A diagonal cell is split into the two time orders, each with
        three vertices.  Thus neither the C kink nor an auxiliary
        continuation is mixed into a linear cell, and a positive table
        remains positive.  Cubic interpolation retains the smooth
        min-time/lag spline and its auxiliary continuation.
        """
        if interp.method != "linear":
            return np.asarray(interp(pts))
        flat = pts.reshape(-1, pts.shape[-1])
        ts = interp.grid[0]
        earlier, later = flat[:, 0], flat[:, 0] + flat[:, 1]
        keep = (
            (earlier >= ts[0])
            & (later <= np.nextafter(ts[-1], np.inf))
            & (later >= earlier)
            & np.all(np.isfinite(flat), axis=1)
        )
        out = np.empty(len(flat), dtype=interp.values.dtype)
        if np.any(~keep):
            out[~keep] = interp(flat[~keep])
        if not np.any(keep):
            return out.reshape(pts.shape[:-1])
        earlier, later = earlier[keep], later[keep]
        i = np.clip(np.searchsorted(ts, earlier, side="right") - 1, 0, len(ts) - 2)
        k = np.clip(np.searchsorted(ts, later, side="right") - 1, 0, len(ts) - 2)
        a = np.clip((later - ts[k]) / (ts[k + 1] - ts[k]), 0.0, 1.0)
        b = np.clip((earlier - ts[i]) / (ts[i + 1] - ts[i]), 0.0, 1.0)
        diagonal = i == k
        s_idx = np.array([i, i, np.where(diagonal, i, i + 1), i + 1])
        t_idx = np.array([k, k + 1, k, k + 1])
        weights = np.array([(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b])
        weights[:, diagonal] = np.array(
            [
                1 - a[diagonal],
                a[diagonal] - b[diagonal],
                np.zeros(np.count_nonzero(diagonal)),
                b[diagonal],
            ]
        )
        # The time coordinates are exact grid vertices.  Gather them
        # directly, interpolating only the remaining spatial axes; asking
        # RGI to rediscover all four time corners multiplies query cost.
        spatial_cells = []
        for axis, grid in enumerate(interp.grid[2:], start=2):
            coordinate = flat[keep, axis]
            if len(grid) == 1:
                spatial_cells.append(((np.zeros(len(coordinate), dtype=int), 1.0),))
                continue
            index = np.clip(
                np.searchsorted(grid, coordinate, side="right") - 1, 0, len(grid) - 2
            )
            fraction = (coordinate - grid[index]) / (grid[index + 1] - grid[index])
            spatial_cells.append(((index, 1 - fraction), (index + 1, fraction)))
        values = np.zeros(weights.shape, dtype=interp.values.dtype)
        for corner in itertools.product(*spatial_cells):
            indices = (s_idx, t_idx - s_idx) + tuple(c[0] for c in corner)
            spatial_weight = 1.0
            for _, weight in corner:
                spatial_weight = spatial_weight * weight
            values += interp.values[indices] * spatial_weight
        out[keep] = np.sum(weights * values, axis=0)
        return out.reshape(pts.shape[:-1])

    def __call__(self, pts):
        pts = np.asarray(pts, dtype=float)
        if pts.ndim == 1:
            pts = pts[None, :]
        t1, t2 = pts[..., 0], pts[..., 1]
        s, u, ge = self._fold(t1, t2)
        spa = pts[..., 2:]
        folded = np.concatenate([s[..., None], u[..., None], spa], axis=-1)
        # Each query needs only its own time-order branch.
        out = np.empty(t1.shape, dtype=self._ge.values.dtype)
        if np.any(ge):
            out[ge] = self._evaluate(self._ge, folded[ge])
        if np.any(~ge):
            out[~ge] = self._evaluate(self._lt, folded[~ge])
        return out

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self._ge, name)


#: The range :func:`_probe_diagonal_cusp` assumes when none is given, and the
#: times it reads, as fractions of the range.
_CUSP_PROBE_SPAN = 3.0
_CUSP_PROBE_TIMES = (0.07, 0.7 / 3.0, 0.52, 2.3 / 3.0, 0.95)
#: Gauss-Legendre nodes per panel of :func:`_graded_rule`, the count
#: :func:`select_graded_levels` compares it with, and the depths it tries,
#: down to ``4e-9`` of the interval.
_GRADED_PANEL_NODES = 8
_GRADED_CHECK_NODES = 12
_GRADED_LEVELS = (12, 16, 20, 24, 28)
#: Where :func:`_c_live_entries` reads the kernel: times on the range, and lags
#: as fractions of it, from the whole range to below the depths above.
_C_LIVE_TIMES = 32
_C_LIVE_LAGS = tuple(4.0**-k for k in range(15))
#: Rounding of one entry of C, as a fraction of the sum of the sizes of its
#: terms, when R mixes components (:func:`_c_rule_difference`).
_C_RULE_ROUNDING = 32.0 * float(np.finfo(float).eps)
#: Rules kept per cache by :meth:`PropagatorCache._c_rule`.  Above the pairs of
#: any table that can be built, which reads its rules cell by cell.
_C_RULE_MEMO_SIZE = 1 << 18
#: The rule of a pair where the kernel feeds no entry of C: there is nothing
#: to converge on, so the count is the starting one.
_ANY_C_RULE = ("gauss_legendre", None)


@lru_cache(maxsize=64)
def _leggauss(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Read-only Gauss-Legendre nodes and weights on ``[-1, 1]``."""
    from numpy.polynomial.legendre import leggauss

    nodes, weights = leggauss(int(n))
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


class _PerNodeCell(Exception):
    """A cell the batched contraction of :func:`_C_value_direct_gl` does not
    take: a κ² that is not an ``(N, N)`` array, or a κ² or R that is not
    finite.  The cell is evaluated node by node instead."""


def _in_order(terms: np.ndarray) -> np.ndarray:
    """``0 + terms[0] + terms[1] + ...``, added left to right as a ``+=``
    loop adds them: ``np.add.accumulate`` is sequential by definition, and
    ``np.sum`` promises no order."""
    start = np.zeros((1, *terms.shape[1:]))
    return np.add.accumulate(np.concatenate((start, terms)), axis=0)[-1]


def _C_value_direct_gl(
    model: "PropagatorModel",
    n1: Any,
    t1: float,
    n2: Any,
    t2: float,
    n_gauss: int = DEFAULT_C_N_GAUSS,
    graded_levels: int | None = None,
    graded_nodes: int = _GRADED_PANEL_NODES,
    term_sizes: bool = False,
) -> np.ndarray:
    """Gauss-Legendre evaluation of the C-propagator 2-D integral.

    Drop-in replacement for the ``scipy.integrate.dblquad`` path inside
    :meth:`PropagatorCache._C_value_direct` for the common case where
    κ² is smooth except for a folded ``|λ1 − λ2|`` cusp on the diagonal
    (the OU / exponential-temporal kernels used in demo1, demo2, and
    the test suite).

    Strategy: rather than apply a single tensor-product Gauss-Legendre
    rule on the rectangle ``[t_min, t1] × [t_min, t2]``, where the
    diagonal cusp ruins polynomial convergence and gives O(10%) error
    even at ``n_gauss=30``, we split the rectangle at ``λ1 = λ2``
    into 2 (square case) or 3 (rectangle case) sub-regions on each of
    which the integrand is smooth, then apply a fixed
    ``n_gauss × n_gauss`` rule per sub-region.

    Sub-regions for ``t_min ≤ λ1 ≤ t1, t_min ≤ λ2 ≤ t2``:

    * Let ``t_d = min(t1, t2)``.  In the square ``[t_min, t_d]²`` the
      diagonal cuts through.

      - Region A (lower triangle): ``λ2 ≤ λ1``, so ``|λ1−λ2|=λ1−λ2``.
      - Region B (upper triangle): ``λ1 ≤ λ2``, so ``|λ1−λ2|=λ2−λ1``.

    * Outside the square (only if ``t1 ≠ t2``):

      - If ``t1 > t2``: Region C is the strip
        ``t_d ≤ λ1 ≤ t1, t_min ≤ λ2 ≤ t2`` where ``λ1 > λ2``.
      - If ``t2 > t1``: Region C is the strip
        ``t_min ≤ λ1 ≤ t1, t_d ≤ λ2 ≤ t2`` where ``λ2 > λ1``.

    Each sub-region is handled by an iterated 1-D Gauss-Legendre rule:
    outer pass over ``λ1`` with ``n_gauss`` nodes, inner pass over
    ``λ2`` with ``n_gauss`` nodes whose interval depends on ``λ1`` for
    the triangular regions.

    For the triangular regions the inner integral has a node-dependent
    upper or lower bound; we map the inner ``[-1, 1]`` GL nodes onto
    that variable interval each time, accumulating the Jacobian.

    The white-noise δ-correlated piece (when ``model.sigma2`` is set)
    is added via a single 1-D Gauss-Legendre pass on
    ``[t_min, min(t1, t2)]``, where the integrand is smooth (no cusp).

    Args:
        model: The propagator model.
        n1, t1, n2, t2: As in :meth:`PropagatorCache._C_value_direct`.
        n_gauss: Number of GL nodes per dimension per sub-region.
        graded_levels: When set, every interval takes the composite rule
            :func:`_graded_rule` at this depth in place of ``n_gauss``
            nodes.  The sub-regions end on the diagonal and at ``t1`` and
            ``t2``, which is where a narrow kernel and a fast rate put all
            of the integral, so the panels shrink towards those features
            with no scale given.
        graded_nodes: Nodes per panel of that rule.
        term_sizes: Integrate ``Σ |R_ai κ_ij R_bj|`` in place of the signed
            sum, the size rounding of an entry of C is relative to.

    Returns:
        ``(N, N)`` array of C values, identical in shape and meaning
        to the dblquad path.
    """
    N = model.n_components
    t_min = model.t_min
    C_mat = np.zeros((N, N))

    if t1 <= t_min or t2 <= t_min:
        # Empty integration domain along at least one axis, so C = 0.
        return C_mat

    # 1-D GL nodes & weights on [-1, 1], reused across sub-regions.
    nodes, weights = _leggauss(n_gauss)

    def _rule(lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
        if graded_levels is not None:
            return _graded_rule(lo, hi, graded_levels, graded_nodes)
        half = 0.5 * (hi - lo)
        return 0.5 * (hi + lo) + half * nodes, half * weights

    # ---- helper: evaluate the R matrix at a single time pair ------ #
    def _r_mat(t_obs: float, lam: float) -> np.ndarray:
        """Return the full ``(N, N)`` response matrix ``R(t_obs, lam)``.

        For ``iso_R`` this is ``R_time * I``.  Otherwise ``R_time`` is
        expected to return the full matrix; note it is **not** in general
        diagonal in component indices; that holds only when the linear
        operator itself is diagonal in the chosen basis.  Using only the
        diagonal here silently corrupts C for any dense drift matrix
        (e.g. ``A = H + lambda`` with ``H = X^T X / N``).
        """
        rt = model.R_time(t_obs, lam)
        if model.iso_R:
            return float(rt) * np.eye(N)
        return np.asarray(rt, dtype=float)

    # ---- helper: evaluate κ² at a single time pair --------------- #
    def _kappa(lam1: float, lam2: float) -> np.ndarray:
        return np.asarray(_kappa2_at(model, n1, lam1, n2, lam2), dtype=float)

    # ---- helper: contract one (R(t1,lam1), kappa, R(t2,lam2)) ----- #
    # triple into the (N, N) per-point integrand value.
    def _integrand_block(
        r1: np.ndarray,
        kmat: np.ndarray,
        r2: np.ndarray,
    ) -> np.ndarray:
        # C_{ab} = sum_{c,d} R_{ac}(t1,l1) kappa_{cd}(l1,l2) R_{bd}(t2,l2)
        #        = [ R1 @ kappa @ R2^T ]_{ab}
        # The transpose is essential and invisible for symmetric R.
        # Do not "simplify" it away.
        kmat = np.asarray(kmat, dtype=float)
        if term_sizes:
            out = np.abs(r1) @ np.abs(kmat) @ np.abs(r2).T
        else:
            out = r1 @ kmat @ r2.T
        if model.diag_C:
            # Project onto the diagonal AFTER contracting, never before.
            return np.diag(np.diag(out))
        return out

    # ---- helper: integrate over a triangular region -------------- #
    # Region of the form  a_lo ≤ λ1 ≤ a_hi,  inner_lo(λ1) ≤ λ2 ≤ inner_hi(λ1).
    # Both ``a_lo, a_hi`` are scalars; ``inner_lo`` and ``inner_hi``
    # are callables of λ1.
    def _gl_region(
        lam1_lo: float,
        lam1_hi: float,
        inner_lo: Callable[[float], float],
        inner_hi: Callable[[float], float],
    ) -> np.ndarray:
        if lam1_hi <= lam1_lo:
            return np.zeros((N, N))
        lam1_pts, outer_w = _rule(lam1_lo, lam1_hi)

        acc = np.zeros((N, N))
        for k, lam1 in enumerate(lam1_pts):
            lo = inner_lo(float(lam1))
            hi = inner_hi(float(lam1))
            if hi <= lo:
                continue
            lam2_pts, inner_w = _rule(lo, hi)

            # A copy: it is kept while R(t2, ·) is called, and an R may
            # fill and return one buffer.
            r1m = np.array(_r_mat(t1, float(lam1)))
            inner_acc = np.zeros((N, N))
            for j, lam2 in enumerate(lam2_pts):
                kmat = _kappa(float(lam1), float(lam2))
                r2m = _r_mat(t2, float(lam2))
                inner_acc += inner_w[j] * _integrand_block(r1m, kmat, r2m)
            acc += outer_w[k] * inner_acc
        return acc

    # ---- the same region under iso_R, the arithmetic batched ----- #
    # R1 = r1 I and R2 = r2 I, so R1 @ κ @ R2^T adds only exact zeros to
    # (r1 κ) r2, entry by entry; that holds for finite values only, and a
    # κ² or R that is not finite takes the node-by-node region.  κ² and R
    # are called at the same nodes in the same order, and both sums run in
    # the order of the loop above, so the value is the same to the bit.
    off_diagonal = ~np.eye(N, dtype=bool)

    def _gl_region_iso(
        lam1_lo: float,
        lam1_hi: float,
        inner_lo: Callable[[float], float],
        inner_hi: Callable[[float], float],
    ) -> np.ndarray:
        if lam1_hi <= lam1_lo:
            return np.zeros((N, N))
        lam1_pts, outer_w = _rule(lam1_lo, lam1_hi)
        terms = []
        for k, lam1 in enumerate(lam1_pts):
            lam1 = float(lam1)
            lo, hi = inner_lo(lam1), inner_hi(lam1)
            if hi <= lo:
                continue
            lam2_pts, inner_w = _rule(lo, hi)
            r1 = float(model.R_time(t1, lam1))
            ks = np.empty((len(lam2_pts), N, N))
            r2s = np.empty(len(lam2_pts))
            for j, lam2 in enumerate(lam2_pts):
                kmat = _kappa(lam1, float(lam2))
                if kmat.shape != (N, N):
                    raise _PerNodeCell
                ks[j] = kmat
                r2s[j] = float(model.R_time(t2, float(lam2)))
            if not (
                np.isfinite(r1) and np.all(np.isfinite(ks)) and np.all(np.isfinite(r2s))
            ):
                raise _PerNodeCell
            if term_sizes:
                blocks = (abs(r1) * np.abs(ks)) * np.abs(r2s)[:, None, None]
            else:
                blocks = (r1 * ks) * r2s[:, None, None]
            if model.diag_C:
                blocks[:, off_diagonal] = 0.0
            terms.append(outer_w[k] * _in_order(inner_w[:, None, None] * blocks))
        return _in_order(np.asarray(terms)) if terms else np.zeros((N, N))

    t_d = min(t1, t2)

    def _regions(region: Callable[..., np.ndarray]) -> np.ndarray:
        out = np.zeros((N, N))
        # Region A: t_min ≤ λ2 ≤ λ1 ≤ t_d  (lower triangle, λ1 > λ2).
        if t_d > t_min:
            out += region(
                t_min,
                t_d,
                inner_lo=lambda _l1: t_min,
                inner_hi=lambda l1: l1,
            )
            # Region B: t_min ≤ λ1 ≤ λ2 ≤ t_d  (upper triangle, λ1 < λ2).
            out += region(
                t_min,
                t_d,
                inner_lo=lambda l1: l1,
                inner_hi=lambda _l1: t_d,
            )

        # Region C: outside the square (only if t1 != t2).
        if t1 > t2:
            # λ1 ∈ [t_d, t1], λ2 ∈ [t_min, t2] = [t_min, t_d].  λ1 > λ2.
            out += region(
                t_d,
                t1,
                inner_lo=lambda _l1: t_min,
                inner_hi=lambda _l1: t_d,
            )
        elif t2 > t1:
            # λ2 ∈ [t_d, t2], λ1 ∈ [t_min, t1] = [t_min, t_d].  λ2 > λ1.
            out += region(
                t_min,
                t_d,
                inner_lo=lambda _l1: t_d,
                inner_hi=lambda _l1: t2,
            )
        return out

    if model.iso_R:
        try:
            C_mat = _regions(_gl_region_iso)
        except _PerNodeCell:
            C_mat = _regions(_gl_region)
    else:
        C_mat = _regions(_gl_region)

    # ---- White-noise δ piece: 1-D GL on [t_min, min(t1, t2)] ----- #
    if model.sigma2 is not None:
        t_upper = t_d
        if t_upper > t_min:
            tau_pts, tau_w = _rule(t_min, t_upper)
            for k, tau in enumerate(tau_pts):
                tau_f = float(tau)
                r1m = np.array(_r_mat(t1, tau_f))  # kept; see above
                r2m = _r_mat(t2, tau_f)
                sig = np.asarray(_sigma2_at(model, n1, tau_f, n2), dtype=float)
                C_mat += tau_w[k] * _integrand_block(r1m, sig, r2m)

    return C_mat


def _graded_rule(
    lo: float, hi: float, levels: int, nodes: int = _GRADED_PANEL_NODES
) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights on ``[lo, hi]`` from panels halved towards both ends.

    The panels are ``levels`` on each side, the smallest ``2^-levels`` of the
    interval, with ``nodes`` Gauss-Legendre nodes each.
    """
    x, w = _leggauss(nodes)
    towards_lo = 2.0 ** -np.arange(levels, 0, -1, dtype=float)
    edges = np.concatenate(([0.0], towards_lo, 1.0 - towards_lo[-2::-1], [1.0]))
    edges = lo + (hi - lo) * edges
    half = 0.5 * np.diff(edges)
    mid = 0.5 * (edges[:-1] + edges[1:])
    return (mid[:, None] + half[:, None] * x).ravel(), (half[:, None] * w).ravel()


def _probe_diagonal_cusp(
    model: "PropagatorModel", positions: tuple, span: float = _CUSP_PROBE_SPAN
) -> bool:
    """Numerically decide whether ``κ²`` has a derivative jump at ``λ1 = λ2``
    on the range of length ``span`` from ``t_min``.

    ``J(h) = |κ(λ, λ+h) + κ(λ, λ−h) − 2 κ(λ, λ)| / h`` is the jump of the
    one-sided derivatives.  For a smooth kernel it is ``O(h)``; for a
    ``|Δ|`` cusp it tends to a constant.  Compare ``h`` and ``h / 4``, at
    the times :data:`_CUSP_PROBE_TIMES`.  A cusp present at none of them is
    not seen.  Entries are judged one by one, each on its own scale, so a
    weak cusp beside a strong smooth entry is seen.
    """
    n1, n2 = positions
    t_min = float(model.t_min)
    unit = float(span) / _CUSP_PROBE_SPAN
    try:
        coarse, fine = [], []
        for f in _CUSP_PROBE_TIMES:
            lam0 = t_min + f * float(span)
            for h, vals in ((1e-3 * unit, coarse), (2.5e-4 * unit, fine)):
                # ``np.array`` copies: a κ² that fills and returns one
                # buffer would otherwise make the three values one array.
                k0 = np.array(_kappa2_at(model, n1, lam0, n2, lam0), dtype=float)
                kp = np.array(_kappa2_at(model, n1, lam0, n2, lam0 + h), dtype=float)
                km = np.array(_kappa2_at(model, n1, lam0, n2, lam0 - h), dtype=float)
                scale = np.maximum(np.abs(k0), 1e-300)
                vals.append(np.abs(kp + km - 2.0 * k0) / h / scale)
    except Exception:
        return True
    j_h, j_h4 = unit * sum(coarse), unit * sum(fine)
    return bool(np.any((j_h4 >= 1e-9) & (j_h4 > 0.5 * j_h)))


def _c_table_entries(n_components: int, diag_C: bool) -> tuple:
    """``(a, b)`` component pairs a C table stores, in storage order.

    Under ``diag_C`` the diagonal ``C_aa`` only; otherwise every ``C_ab``,
    row-major, which is what an off-diagonal C (a dense R, a
    component-mixing κ², a matrix σ²) needs.
    """
    n = int(n_components)
    if diag_C:
        return tuple((a, a) for a in range(n))
    return tuple((a, b) for a in range(n) for b in range(n))


#: Times, as fractions of a table's range, at which :func:`_noise_reads_alike`
#: reads κ² and σ².  Irregular, so that no periodic time dependence vanishes
#: at all of them.
_NOISE_PROBE_TIMES = (0.03, 0.17, 0.38, 0.59, 0.76, 0.97)


def _noise_reads_alike(
    model: "PropagatorModel", pair: tuple, other: tuple, t_range: tuple
) -> bool:
    """Whether κ² and σ² take the same values at two pairs of positions, each
    entry to 1e-12 of its own size, at every pair of
    :data:`_NOISE_PROBE_TIMES` in ``t_range``.

    A dependence on position that vanishes at all of those times is not seen.
    """
    lo, hi = t_range
    times = [lo + f * (hi - lo) for f in _NOISE_PROBE_TIMES]

    def alike(u: Any, v: Any) -> bool:
        u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
        return bool(np.all(np.abs(u - v) <= 1e-12 * np.maximum(np.abs(u), np.abs(v))))

    for t1, t2 in itertools.product(times, times):
        # A copy: a κ² may fill and return one buffer.
        first = np.array(_kappa2_at(model, pair[0], t1, pair[1], t2), dtype=float)
        if not alike(first, _kappa2_at(model, other[0], t1, other[1], t2)):
            return False
    if model.sigma2 is None:
        return True
    return all(
        alike(
            np.array(_sigma2_at(model, pair[0], t, pair[1]), dtype=float),
            _sigma2_at(model, other[0], t, other[1]),
        )
        for t in times
    )


#: Fractions of a table's range at which :func:`_c_probe_cells` reads the
#: diagonal, between the extreme cells, and the lags of its off-diagonal cells.
#: Irregular, as :data:`_NOISE_PROBE_TIMES`.
_C_PROBE_DIAGONAL = (0.08, 0.17, 0.26, 0.35, 0.43, 0.58, 0.67, 0.76, 0.84, 0.92)
_C_PROBE_LAGS = (0.2, 0.35, 0.7)
_C_EXTREME_CELLS = 5


def select_gl_node_count(
    model: "PropagatorModel",
    t_max: float,
    *,
    n_start: int = 20,
    n_max: int = 110,
    tol: float = 1e-8,
    growth: float = 1.5,
    positions: tuple | None = None,
    unchecked: set | None = None,
    live: tuple | None = None,
    raised: list | None = None,
) -> int | None:
    """Smallest Gauss-Legendre node count that is converged for a table
    reaching ``t_max``, or ``None`` if none up to ``n_max`` is.

    A fixed-node tensor-product rule loses accuracy as the integrand's
    exponential rates times the interval length grow: for the demo1
    kernel (``γ = 1``, ``σ_t = 0.3``) ``n_gauss = 20`` is at 1e-12 for
    ``t_max = 15`` but 1e-5 at 30 and 2e-2 at 100.  Rather than trust a
    number, this probes the cells of :func:`_c_probe_cells`, the extreme ones
    first since most kernels fail there, and accepts ``n`` when the rule agrees
    with its ``growth × n`` refinement to ``tol`` relative at every probe
    (:func:`_c_rule_difference` says how).  This is the nested-rule
    convergence test every adaptive quadrature runs, applied once per
    table instead of per cell; it costs a few tens of GL evaluations, each
    the price of one table cell.

    ``positions`` is the pair ``(n1, n2)`` to probe at, the origin by default.
    ``unchecked`` collects the entries the accepted comparison could not
    check (:func:`_c_rule_difference`).

    Returns ``None``, meaning "try :func:`select_graded_levels`", when the
    cap is reached or the kernel raises on the probe positions.  The cap is
    ``n_max``, or the refinement of ``n_start`` when that is larger, so a
    start of any size is tested at least once.

    A feature narrower than the node spacing that neither rule lands on can
    leave both rules with the same wrong value, on the diagonal too: a spike
    narrower than about
    ``1e-8`` of the interval leaves ``n = 20`` and ``n = 30`` in agreement, so
    the :func:`select_graded_levels` that would resolve it is never reached.
    White noise is the spelling for such a kernel.

    ``t_max`` is converted to a float first: the probe cells pass it to
    ``R_time`` at every node, and a 0-d array would let an ``R_time`` that
    writes into its time argument move the caller's ``t_max``.

    ``live`` is :func:`_c_live_entries` at these positions up to ``t_max``,
    when the caller has it.  A cell is evaluated once per count: the second
    pass reads the extreme cells of the first.  ``raised`` collects the
    exception that ended the search, when one did.
    """
    import math

    t_max = float(t_max)
    t_min = float(model.t_min)
    if t_max <= t_min:
        return int(n_start)
    probes = _c_probes(t_min, t_max, positions)
    values: dict[tuple[int, int], np.ndarray] = {}

    def value(i: int, n: int) -> np.ndarray:
        if (i, n) not in values:
            values[i, n] = _C_value_direct_gl(model, *probes[i], n_gauss=n)
        return values[i, n]

    def refined(count: int) -> int:
        return max(count + 2, int(math.ceil(growth * count)))

    cap = max(int(n_max), refined(int(n_start)))

    def converged(n: int, n_cells: int, report: set | None) -> int | None:
        check = _c_rule_check(model, probes[:n_cells], live=live, n_gauss=n)
        while True:
            n_ref = refined(n)
            if n_ref > cap:
                return None
            vals = [value(i, n) for i in range(n_cells)]
            refs = [value(i, n_ref) for i in range(n_cells)]
            worst = _c_rule_difference(vals, refs, check, tol, report)
            if worst is None:
                return None
            if worst <= tol:
                return n
            n = n_ref

    try:
        n = converged(int(n_start), _C_EXTREME_CELLS, None)
        return None if n is None else converged(n, len(probes), unchecked)
    except Exception as exc:
        if raised is not None:
            raised.append(exc)
        return None


def select_graded_levels(
    model: "PropagatorModel",
    t_max: float,
    *,
    tol: float = 1e-8,
    positions: tuple | None = None,
    unchecked: set | None = None,
    live: tuple | None = None,
    raised: list | None = None,
) -> int | None:
    """Smallest depth of :func:`_graded_rule` that is converged for a table
    reaching ``t_max``, or ``None`` if none is.

    The rule of last resort under ``c_method='auto'``, for a kernel or a rate
    so narrow against the horizon that :func:`select_gl_node_count` gives up.
    The test is the one that function runs, in two steps.  Depths of
    :data:`_GRADED_LEVELS` are compared first, which settles the panels at
    the ends.  Those depths share every interior panel, so their agreement
    says nothing about a kink or a ridge inside the interval: the depth found
    is then compared with the same panels at :data:`_GRADED_CHECK_NODES`
    nodes, and a disagreement there returns ``None``, since more depth adds
    nothing in the interior.  An integral that comes out 0 where the kernel
    feeds that entry (:func:`_c_live_entries`) is a feature no depth has
    reached yet, so the search goes on.

    ``positions``, ``unchecked``, ``live`` and ``raised`` are as in
    :func:`select_gl_node_count`, and a cell is likewise evaluated once per
    rule.
    """
    t_max = float(t_max)
    probes = _c_probes(float(model.t_min), t_max, positions)
    memo: dict[tuple[int, int, int], np.ndarray] = {}

    def values(
        n_cells: int, levels: int, nodes: int = _GRADED_PANEL_NODES
    ) -> list[np.ndarray]:
        for i in range(n_cells):
            if (i, levels, nodes) not in memo:
                memo[i, levels, nodes] = _C_value_direct_gl(
                    model, *probes[i], graded_levels=levels, graded_nodes=nodes
                )
        return [memo[i, levels, nodes] for i in range(n_cells)]

    def converged(n_cells: int, start: int, report: set | None) -> int | None:
        depths = [d for d in _GRADED_LEVELS if d >= start]
        check = _c_rule_check(
            model, probes[:n_cells], live=live, graded_levels=depths[0]
        )
        vals = values(n_cells, depths[0])
        for levels, finer in zip(depths[:-1], depths[1:]):
            refs = values(n_cells, finer)
            worst = _c_rule_difference(vals, refs, check)
            if worst is not None and worst <= tol:
                checked = values(n_cells, levels, _GRADED_CHECK_NODES)
                worst = _c_rule_difference(vals, checked, check, tol, report)
                return levels if worst is not None and worst <= tol else None
            vals = refs
        return None

    try:
        levels = converged(_C_EXTREME_CELLS, _GRADED_LEVELS[0], None)
        return None if levels is None else converged(len(probes), levels, unchecked)
    except Exception as exc:
        if raised is not None:
            raised.append(exc)
        return None


def _c_probe_cells(t_min: float, t_max: float) -> list[tuple[float, float]]:
    """The cells a rule of C is tested at, for a table reaching ``t_max``.

    First the :data:`_C_EXTREME_CELLS` extreme cells: the deep corner, its
    mid-table neighbours and the thinnest strip.  Then the diagonal at :data:`_C_PROBE_DIAGONAL`, since a cell
    sees κ² only within a few decay times of ``R`` before it.  The last row and
    column at the lags of :data:`_C_PROBE_LAGS`, on both sides, since κ² need
    not be symmetric in time.

    A diagonal cell sees a feature of κ² down to the tolerance over
    ``log(1 / tol) / (2 rate)``, so the diagonal covers a table up to a rate
    times its range of about 100.  Beyond that, or for a feature away from the
    diagonal at positions where κ² is not a covariance of one field, an
    interior feature can go unread.
    """
    span = t_max - t_min
    h = span / 60.0
    mid = t_min + 0.5 * span
    extreme = [
        (t_max, t_max),
        (t_max, mid),
        (mid, mid),
        (t_max, t_min + h),
        (t_min + h, t_min + h),
    ]
    lags = [(t_max, t_max - lag * span) for lag in _C_PROBE_LAGS]
    return [
        *extreme,
        *((t_min + f * span, t_min + f * span) for f in _C_PROBE_DIAGONAL),
        *lags,
        *((t2, t1) for t1, t2 in [*lags, (t_max, mid), (t_max, t_min + h)]),
    ]


def _c_probes(t_min: float, t_max: float, positions: tuple | None) -> list[tuple]:
    """``(n1, t1, n2, t2)`` at the probe cells, at one pair of positions."""
    n1, n2 = (np.asarray(0.0), np.asarray(0.0)) if positions is None else positions
    return [(n1, t1, n2, t2) for t1, t2 in _c_probe_cells(t_min, t_max)]


def _c_rule_horizon(t_min: float, t: float) -> float:
    """The horizon a rule for a cell reaching ``t`` is worked out for:
    ``t_min`` plus the first power of 2 above ``t - t_min``.  Every cell below
    it shares the answer, and the answer depends on nothing but ``t``."""
    import math

    exponent = math.frexp(t - t_min)[1]
    return max(t, t_min + math.ldexp(1.0, min(exponent, 1023)))


def _position_key(x: Any) -> tuple:
    """A hashable stand-in for a position, for the memos on a cache."""
    arr = np.asarray(x, dtype=float)
    return arr.shape, arr.tobytes()


def _pair_key(n1: Any, n2: Any) -> tuple:
    """A hashable stand-in for a pair of positions."""
    return _position_key(n1), _position_key(n2)


#: Characters of a user callable's error a warning quotes.
_MAX_ERROR_CHARS = 500


def _error_text(exc: BaseException) -> str:
    """``Type: message`` of an error from a user callable, cut to
    :data:`_MAX_ERROR_CHARS` characters, since its message can be any size.
    The type alone when its ``str`` raises."""
    try:
        text = f"{type(exc).__name__}: {exc}"
    except Exception:  # noqa: BLE001 (a user exception's __str__)
        return type(exc).__name__
    if len(text) <= _MAX_ERROR_CHARS:
        return text
    return f"{text[:_MAX_ERROR_CHARS]}... ({len(text)} characters)"


def _combined_c_rule(rules: Any, n_start: int) -> tuple[str, int | None]:
    """What several pairs' rules come to, for a label.

    The graded rule, at the largest depth, when any pair takes it, else
    Gauss-Legendre at the largest count.  ``dblquad`` when every pair takes
    it, and ``'auto'`` when only some do, since no one rule then describes
    the table.  Each pair uses its own rule whatever this says.
    """
    methods = [method for method, _ in rules]
    if "dblquad" in methods:
        return ("dblquad", None) if set(methods) == {"dblquad"} else ("auto", None)
    method = "graded" if "graded" in methods else "gauss_legendre"
    return method, max(n or n_start for m, n in rules if m == method)


def _c_live_entries(
    model: "PropagatorModel", n1: Any, n2: Any, t_hi: float
) -> tuple[np.ndarray, bool]:
    """``(N, N)`` mask of the entries of C the kernel feeds at ``(n1, n2)``,
    and whether R mixes components.

    An entry of C can only be 0 for a real reason when it is off here, so
    this is what tells a C that is 0 from a feature no rule has reached.  κ²
    (and σ²) are read at :data:`_C_LIVE_TIMES` times on ``[t_min, t_hi]`` and
    at lags from :data:`_C_LIVE_LAGS` around each, and R at those lags gives
    which components an entry reaches through.

    A kernel that reads 0 at all of them counts as 0, so a pulse in absolute
    time narrower than about 1/32 of the range and missed by every rule too
    comes out 0.  A kernel that raises feeds everything: the caller finds out
    what the rule does with it.
    """
    N = int(model.n_components)
    t_lo = float(model.t_min)
    span = t_hi - t_lo
    if not span > 0.0:
        return np.zeros((N, N), dtype=bool), False
    times = t_lo + span * (np.arange(_C_LIVE_TIMES) + 0.5) / _C_LIVE_TIMES
    lags = span * np.asarray(_C_LIVE_LAGS)
    kernel = np.zeros((N, N), dtype=bool)
    reach = np.eye(N, dtype=bool)
    try:
        for s in times:
            s = float(s)
            kernel |= _nonzero_entries(_kappa2_at(model, n1, s, n2, s), N)
            if model.sigma2 is not None:
                kernel |= _nonzero_entries(_sigma2_at(model, n1, s, n2), N)
            for lag in lags:
                lag = float(lag)
                for other in (s - lag, s + lag):
                    if t_lo <= other <= t_hi:
                        kernel |= _nonzero_entries(
                            _kappa2_at(model, n1, s, n2, other), N
                        )
                        kernel |= _nonzero_entries(
                            _kappa2_at(model, n1, other, n2, s), N
                        )
                if not model.iso_R and t_lo <= s - lag:
                    reach |= _nonzero_entries(model.R_time(s, s - lag), N)
    except Exception:  # noqa: BLE001 (a user callable; see the docstring)
        return np.ones((N, N), dtype=bool), not model.iso_R
    live = (reach.astype(float) @ kernel.astype(float) @ reach.T.astype(float)) > 0.0
    mixes = bool(np.any(reach & ~np.eye(N, dtype=bool)))
    return (live & np.eye(N, dtype=bool) if model.diag_C else live), mixes


def _nonzero_entries(value: Any, N: int) -> np.ndarray:
    """``(N, N)`` mask of where ``value``, a matrix or a number, is not 0;
    not finite counts as not 0."""
    return np.broadcast_to(np.asarray(value, dtype=float) != 0.0, (N, N))


def _c_rule_check(
    model: "PropagatorModel", probes: list, live: tuple | None = None, **rule: Any
) -> tuple[np.ndarray, np.ndarray]:
    """What the comparison of two rules needs at ``probes``, the cells of one
    pair of positions: the mask of :func:`_c_live_entries`, and the size of the
    terms of each entry at each cell under ``rule`` (keywords of
    :func:`_C_value_direct_gl`), or zeros when R does not mix components and no
    entry adds up across them.

    ``live`` is what :func:`_c_live_entries` returns for these positions up to
    the latest time of ``probes``, when the caller has it.
    """
    n1, n2 = probes[0][0], probes[0][2]
    t_hi = max(max(float(p[1]), float(p[3])) for p in probes)
    N = int(model.n_components)
    live, mixes = _c_live_entries(model, n1, n2, t_hi) if live is None else live
    sizes = (
        np.stack(
            [_C_value_direct_gl(model, *p, term_sizes=True, **rule) for p in probes]
        )
        if mixes
        else np.zeros((len(probes), N, N))
    )
    return live, sizes


def _c_rule_difference(
    vals: list,
    refs: list,
    check: tuple[np.ndarray, np.ndarray],
    tol: float = 0.0,
    unchecked: set | None = None,
) -> float | None:
    """Worst relative difference of two rules over the probe cells of one pair
    of positions, or ``None`` when they say nothing about C.

    Each entry is judged on its own scale, with cells below ``1e-6`` of that
    scale judged against that floor: a component far weaker than another is a
    separate integral, and its error is not small because the other is large.
    When R mixes components an entry is a sum over them that can cancel, so a
    difference within rounding of the size of that entry's own terms does not
    count.  A term of another entry is no part of this one's rounding, whatever
    R does elsewhere.

    That allowance is what can be checked at all.  When the rules agree to
    ``tol`` and the allowance is above ``tol`` times an entry, the entry
    ``(a, b)`` goes into ``unchecked``.

    ``None`` when a value is not finite, or when an entry the kernel feeds
    (``check``) is 0 under both rules, since that says only that neither rule
    reached it.  An entry the kernel does not feed is 0 and needs no rule.  The
    term sizes are read once, under the first rule tried: a size that rule
    misses gives less allowance, so the search goes on.
    """
    live, sizes = check
    v = np.stack([np.asarray(x, dtype=float) for x in vals])
    r = np.stack([np.asarray(x, dtype=float) for x in refs])
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(r))):
        return None
    scale = np.max(np.maximum(np.abs(v), np.abs(r)), axis=0)
    if np.any(live & (scale == 0.0)):
        return None
    seen = scale > 0.0
    if not np.any(seen):
        return 0.0
    floor = np.maximum(np.abs(r), 1e-6 * scale)
    noise = _C_RULE_ROUNDING * sizes
    excess = np.maximum(np.abs(v - r) - noise, 0.0)
    worst = float(np.max((excess / np.where(seen, floor, 1.0))[:, seen]))
    if unchecked is not None and worst <= tol:
        hidden = seen & np.any(noise > tol * floor, axis=0)
        unchecked.update((int(a), int(b)) for a, b in zip(*np.nonzero(hidden)))
    return worst


def _causal_lower_bound_sources(
    spatial,
    int_vars,
    external_times: dict,
    t_min: float,
    swept: tuple = (),
    orderings=None,
) -> tuple[dict, dict]:
    """Lower limits on internal times from *external* response legs, split
    into a constant part and a *variable* part.

    A time ordering ``(earlier, later)`` from an R propagator can be expressed
    as an **upper** bound only when ``earlier`` is itself an integration
    variable (:func:`_causal_bounds`).  When ``earlier`` is an **external**
    point and ``later`` is internal, ``t_later >= t_earlier`` is a **lower**
    bound.  That case arises whenever the observable itself carries a
    response leg, since ``<phi(u) psi(y)> = R(u, y)`` forces ``t_u >= t_y``.

    ``external_times`` maps each **fixed** external point to its time, and
    ``swept`` names the externals the caller is sampling (``integrate_over``).
    An ordering from a swept external is still a lower bound, but its value
    is only known per sample, so it is reported as a *source name* rather
    than a number.

    Returns ``(const, sources)`` where

    * ``const[v]`` is the constant lower limit for internal variable ``v``
      with at least one such bound (``t_min`` folded in), and
    * ``sources[v]`` is a tuple of names drawn from ``swept`` that also
      bound ``v`` from below, so the caller can take a per-sample
      ``max(const[v], *[t(s) for s in sources[v]])``.

    Only names in ``swept`` become sources.  A point that is in neither
    ``external_times`` nor ``swept`` (an aliased leg of an ``equal_time``
    non-local vertex, say) is skipped exactly as before, so the caller
    never receives a name it has no column for.

    Every sampler in this module draws the integrated externals *before*
    any internal variable, and ``nquad`` places them outside every internal
    variable in ``all_vars``, so a source's value is always available by
    the time the bound is needed.

    ``orderings`` defaults to the diagram's causal structure.  The
    integrators that split the domain pass their own, the causal structure
    plus the orderings the split added: ``u ≥ t_x`` and ``u ≤ v`` make
    ``v ≥ t_x`` too, and only the caller knows the second.  Left out, the
    piece keeps ``v`` down to ``t_min`` and ``u``'s interval ``[t_x, v]``
    collapses to zero width below ``t_x``, a jump in ``v`` where the
    integrand had a weaker kink before the split, which costs more than
    the split buys (measured on demo 7's ``integrate_over={'x'}`` channel:
    3.9e-04 at 16 nodes against 4.6e-07 unsplit, and 1.7e-09 with the
    bound carried).
    """
    int_set = set(int_vars)
    swept_set = set(swept)
    orderings = spatial.time_orderings if orderings is None else orderings
    lowers: dict = {}
    sources: dict = {}
    # Direct constraints: (external earlier) -> (internal later).
    for earlier, later in orderings:
        if later in int_set and earlier not in int_set:
            t_e = external_times.get(earlier)
            if t_e is None:
                if earlier in swept_set:
                    sources.setdefault(later, set()).add(earlier)
            else:
                lowers[later] = max(lowers.get(later, t_min), float(t_e))

    # TRANSITIVE CLOSURE along internal edges.  A chain
    # ``ext -> v1 -> v2`` implies ``t_v2 >= t_v1 >= t_ext``, but only ``v1``
    # is named in a direct (external, internal) ordering.  Leaving ``v2`` at
    # ``t_min`` lets it range below ``t_ext``, at which point ``v1``'s interval
    # ``[t_ext, t_v2]`` inverts, nquad then integrates backwards and returns
    # a negative volume.  The internal orderings form a DAG, so this fixpoint
    # terminates; it is cheap because diagrams have very few vertices.
    # Variable sources propagate along the same edges and for the same
    # reason.
    changed = True
    while changed:
        changed = False
        for earlier, later in orderings:
            if earlier in int_set and later in int_set:
                lo_e = lowers.get(earlier)
                if lo_e is not None and lo_e > lowers.get(later, t_min):
                    lowers[later] = lo_e
                    changed = True
                src_e = sources.get(earlier)
                if src_e and not src_e <= sources.get(later, set()):
                    sources.setdefault(later, set()).update(src_e)
                    changed = True
    return lowers, {v: tuple(sorted(s)) for v, s in sources.items()}


def _resolve_external_times(
    spatial,
    ext_fixed,
    lambda_f: float,
    external_times=None,
    integrate_over=(),
) -> tuple[dict, float]:
    """``{fixed external point -> its time}`` plus the global time ceiling.

    ``lambda_f`` plays two distinct roles in this module and only one of them
    generalises:

    * the **sweep limit** for an integrated external, and the default upper
      bound for an internal variable with no causal parent; these stay tied
      to ``lambda_f`` (raised to the ceiling below when some external is
      pinned later than it), and
    * the **time of a fixed external point**, which ``external_times`` now
      overrides per point.  Conflating the two is why unequal external times
      were unreachable: an observable such as ``R(t, t')`` needs its two legs
      at different times, and every production integrator pinned them both.

    ``external_times=None`` reproduces the old behaviour exactly.

    Returns ``(times, ceiling)`` where ``ceiling = max(lambda_f, *times)``.
    An internal variable with no causal parent may still precede an external
    pinned beyond ``lambda_f``, so the default upper bound must cover it.

    Raises:
        ValueError: for an unknown point name, or for a point that is both
            pinned here and swept via ``integrate_over``.
    """
    if external_times:
        known = set(spatial.external_points)
        unknown = sorted(set(external_times) - known)
        if unknown:
            raise ValueError(
                f"external_times names unknown external point(s) {unknown}; "
                f"this diagram's external points are {sorted(known)}."
            )
        clash = sorted(set(external_times) & set(integrate_over or ()))
        if clash:
            raise ValueError(
                f"external point(s) {clash} appear in BOTH external_times and "
                f"integrate_over -- a point cannot be pinned at a fixed time "
                f"and swept at the same time."
            )
    et = external_times or {}
    times = {v: float(et.get(v, lambda_f)) for v in ext_fixed}
    ceiling = max([float(lambda_f), *times.values()]) if times else float(lambda_f)
    return times, ceiling


def _causal_reachability(time_orderings) -> dict:
    """``{node: set of nodes strictly later than it}``, transitively closed.

    Closing over the WHOLE ordering graph, not just edges whose endpoints are
    of one kind, is what makes the constraint table exhaustive.  Enumerating
    edge kinds by endpoint type ({fixed-ext, swept-ext, internal}^2) and
    handling each directly is how the branch previously missed three cells:
    swept -> internal -> swept, fixed-ext -> swept, and swept -> fixed-ext.
    """
    succ: dict = {}
    for e, l in time_orderings:
        succ.setdefault(e, set()).add(l)
        succ.setdefault(l, set())
    changed = True
    while changed:  # the ordering graph is a small DAG
        changed = False
        for n in list(succ):
            grown = set(succ[n])
            for m in succ[n]:
                grown |= succ.get(m, set())
            if grown != succ[n]:
                succ[n] = grown
                changed = True
    return succ


def _swept_external_order(
    spatial,
    ext_integrated,
    fixed_times=None,
    lambda_f=None,
    t_min=0.0,
    orderings=None,
) -> tuple:
    """Causal constraints ON the swept externals themselves.

    A swept external is drawn freely over ``[t_min, lambda_f]`` unless some
    ordering constrains it.  Three kinds do, and none is carried by
    ``_causal_lower_bound_sources`` (which needs the LATER endpoint internal)
    or by the parents of :func:`_causal_bounds` (which need the EARLIER one
    internal):

    * **swept -> swept**, directly or through internal vertices.  Both give
      ``t_later > t_earlier``; the mediated case is why this works off the
      transitive closure rather than the raw edge list.
    * **fixed-ext -> swept**: a constant lower bound at the fixed point's time.
    * **swept -> fixed-ext**: a constant upper bound at that time.

    Theta keeps the VALUE right in every case (the integrand vanishes on the
    unconstrained part), so these are quadrature concerns.  But they are the
    same jump-inside-the-domain that costs ``gauss_legendre`` its spectral
    convergence (measured 22% and 29% at the library default ``n_gauss=8``).

    Returns ``(order, lowers, const_lo, const_hi)``: the swept externals
    topologically sorted so every predecessor is drawn first, the swept
    predecessors of each, and its constant bounds.  With no constraint at all
    the order is the caller's own and the dicts are empty, so sampling stays
    bit-identical.

    ``orderings`` defaults to the diagram's causal structure.  The
    integrators that split the time domain pass their own, the causal
    structure plus the orderings the split added, so that an extra
    swept-to-swept ordering (which :func:`_kink_pairs` pairs and
    :func:`_kink_orientations` chooses) reaches the sampler.
    """
    ext_set = set(ext_integrated)
    orderings = spatial.time_orderings if orderings is None else orderings
    succ = _causal_reachability(tuple(orderings))
    fixed_times = fixed_times or {}

    lowers: dict = {}
    const_lo: dict = {}
    const_hi: dict = {}
    for v in ext_integrated:
        pre = [u for u in succ if v in succ.get(u, ()) and u != v]
        sw = [u for u in pre if u in ext_set]
        if sw:
            lowers[v] = sorted(sw)
        fx = [fixed_times[u] for u in pre if u in fixed_times]
        if fx:
            const_lo[v] = max([float(t_min)] + [float(t) for t in fx])
        post_fx = [fixed_times[u] for u in succ.get(v, ()) if u in fixed_times]
        if post_fx and lambda_f is not None:
            const_hi[v] = min([float(lambda_f)] + [float(t) for t in post_fx])

    if not lowers:
        return list(ext_integrated), {}, const_lo, const_hi

    # Kahn topological sort, stable in the caller's order.
    _refuse_a_time_loop(list(orderings))
    remaining = list(ext_integrated)
    placed: list = []
    placed_set: set = set()
    while remaining:
        ready = [
            v for v in remaining if all(src in placed_set for src in lowers.get(v, ()))
        ]
        for v in ready:
            placed.append(v)
            placed_set.add(v)
            remaining.remove(v)
    return placed, lowers, const_lo, const_hi


class _TimeBound(NamedTuple):
    """The range of one time variable of a causal domain.

    ``lo`` and ``hi`` are its constant limits; ``lower`` and ``upper`` name
    variables, drawn before it, that bound it too.
    """

    var: str
    lo: float
    hi: float
    lower: tuple[str, ...]
    upper: tuple[str, ...]


def _causal_bounds(
    spatial,
    internal,
    swept,
    fixed_times: dict,
    lambda_f: float,
    t_ceiling: float,
    t_min: float,
    orderings=None,
    cuts: tuple[dict, dict] = ({}, {}),
) -> list[_TimeBound]:
    """The causal domain of a diagram: one :class:`_TimeBound` per variable,
    in draw order, the swept externals first.

    An ordering ``(earlier, later)`` bounds an internal ``earlier`` from
    above by ``later``: a sampled variable, or a fixed external at its own
    time, or ``t_ceiling`` with no parent.  Lower limits come from
    :func:`_causal_lower_bound_sources`, the swept externals' own limits
    from :func:`_swept_external_order`.  ``orderings`` defaults to the
    diagram's causal structure, and ``cuts`` holds the ``(upper, lower)``
    constant limits of :func:`_applied_cut_bounds`.

    Every integrator reads the domain from here: the samplers through
    :func:`_causal_times`, ``nquad`` through :func:`_nquad_ranges`.
    """
    orderings = spatial.time_orderings if orderings is None else orderings
    cut_upper, cut_lower = cuts
    sampled = set(internal) | set(swept)
    draw_order, sw_lowers, sw_lo, sw_hi = _swept_external_order(
        spatial, swept, fixed_times, lambda_f, t_min, orderings=orderings
    )
    lowers, lower_srcs = _causal_lower_bound_sources(
        spatial, internal, fixed_times, t_min, swept=swept, orderings=orderings
    )
    bounds = []
    for name in draw_order:
        lo = max(float(t_min), float(sw_lo.get(name, t_min)))
        hi = float(sw_hi.get(name, lambda_f))
        bounds.append(
            _TimeBound(
                name,
                max(lo, cut_lower.get(name, lo)),
                min(hi, cut_upper.get(name, hi)),
                tuple(sw_lowers.get(name, ())),
                (),
            )
        )
    parents: dict[str, list[str]] = defaultdict(list)
    for earlier, later in orderings:
        if earlier in internal:
            parents[earlier].append(later)
    for var in internal:
        upper = [p for p in parents.get(var, ()) if p in sampled]
        limits = [
            fixed_times.get(p, lambda_f)
            for p in parents.get(var, ())
            if p not in sampled
        ]
        if var in cut_upper:
            limits.append(cut_upper[var])
        lo = float(lowers.get(var, t_min))
        bounds.append(
            _TimeBound(
                var,
                max(lo, cut_lower.get(var, lo)),
                min([t_ceiling, *limits]),
                lower_srcs.get(var, ()),
                tuple(upper),
            )
        )
    return bounds


def _causal_times(
    bounds: list[_TimeBound], u: np.ndarray, columns: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Map samples ``u`` of the unit cube onto the domain ``bounds``.

    ``columns`` names the column of each variable.  Returns ``(times,
    jacobians)``.  A sample whose interval is empty sits at the lower limit
    with Jacobian 0.
    """
    times = np.empty_like(u)
    jacobians = np.ones(len(u))
    for var, lo, hi, lower, upper in bounds:
        col = columns[var]
        for src in upper:
            hi = np.minimum(hi, times[:, columns[src]])
        for src in lower:
            lo = np.maximum(lo, times[:, columns[src]])
        width = hi - lo
        valid = width > 0
        times[:, col] = np.where(valid, lo + u[:, col] * width, lo)
        jacobians = np.where(valid, jacobians * width, 0.0)
    return times, jacobians


def _batch_time_reader(
    times: np.ndarray,
    columns: dict[str, int],
    fixed_times: dict,
    fallback: float,
    equal_time_aliases,
) -> Callable[[str], np.ndarray]:
    """``label -> (n_samples,)`` times, read-only (:func:`_read_only_times`).

    A sampled variable reads its column of ``times``, any other label its
    fixed time, or ``fallback``.  A leg of an equal-time vertex and an
    R-absorbed leg read their representative.
    """
    n_samples = len(times)
    _read_only_times(times)
    fixed = {
        v: _read_only_times(np.full(n_samples, t)) for v, t in fixed_times.items()
    }
    other = _read_only_times(np.full(n_samples, fallback))
    alias = dict(equal_time_aliases or ())

    def time_of(label: str) -> np.ndarray:
        label = alias.get(label, label)
        col = columns.get(label)
        return times[:, col] if col is not None else fixed.get(label, other)

    return time_of


def _nquad_ranges(bounds: list[_TimeBound], quad_vars: list[str]) -> list:
    """``ranges`` for ``scipy.integrate.nquad`` over ``quad_vars``, innermost
    first: ``(lo, hi)``, or a callable for a variable bounded by another.

    scipy calls ``ranges[i]`` with the values of the OUTER variables
    ``quad_vars[i+1:]``, so a source of ``quad_vars[i]`` sits at offset
    ``index(src) - i - 1`` of the arguments.  An interval that has collapsed
    returns ``(lo, lo)``: scipy would integrate ``hi < lo`` backwards and
    return a negative volume.
    """
    by_var = {b.var: b for b in bounds}

    def make_bound(bound: _TimeBound, cur_i: int) -> Callable:
        def bound_func(*later_args: float) -> tuple[float, float]:
            def outer(src: str, default: float) -> float:
                j = quad_vars.index(src) - cur_i - 1
                return later_args[j] if 0 <= j < len(later_args) else default

            hi = min([bound.hi] + [outer(s, bound.hi) for s in bound.upper])
            lo = max([bound.lo] + [outer(s, bound.lo) for s in bound.lower])
            return (lo, hi if hi > lo else lo)

        return bound_func

    ranges: list = []
    for i, var in enumerate(quad_vars):
        bound = by_var[var]
        if bound.lower or bound.upper:
            ranges.append(make_bound(bound, i))
        else:
            ranges.append((bound.lo, max(bound.lo, bound.hi)))
    return ranges


def _batch_reality_sizes(re_: np.ndarray, im_: np.ndarray) -> tuple[float, float]:
    """``(scale, worst)``: the largest ``|Re|`` and ``|Im|`` of a batch."""
    scale = float(np.max(np.abs(re_))) if re_.size else 0.0
    worst = float(np.max(np.abs(im_))) if im_.size else 0.0
    return scale, worst


def _batch_reality_error(
    worst: float, scale: float, e_psi: int, where: str
) -> ValueError:
    return ValueError(
        f"Diagram integrand batch{where} has a non-negligible imaginary part "
        f"after the i**E_psi rotation (E_psi={e_psi}): worst |Im| = {worst:.3e} "
        f"against |Re| scale {scale:.3e}.  Check the coupling's phase "
        f"convention: the MSR reality theorem gives i**(-E_psi) x real only "
        f"when each vertex coupling carries (+-i)**n_psi(v)."
    )


def _real_batch_or_raise(values, e_psi: int = 0, *, where: str = "") -> np.ndarray:
    """Array counterpart of :func:`_real_or_raise`.

    The dynamic-coupling batch sites took a bare ``np.real(...)`` after the
    ``i**E_psi`` rotation, so a mis-specified action (a real coupling where
    the MSR convention wants an imaginary one, say) silently lost its
    imaginary part instead of being reported.  That is a hole in the reality
    projection on precisely the feature this branch adds.

    Vectorised so a per-sample batch is checked in one pass: the batch is
    one value of :func:`_residue_negligible`, with its largest ``|Re|`` as
    the scale, the ``scale`` argument of the scalar helper.  Judging element
    by element would reject legitimate near-zero samples inside an otherwise
    healthy batch.
    """
    z = np.asarray(values) * (1j ** int(e_psi))
    re_ = np.real(z)
    im_ = np.imag(z)
    if not np.any(im_):
        return re_
    scale, worst = _batch_reality_sizes(re_, im_)
    if _residue_negligible(worst, scale):
        return re_
    raise _batch_reality_error(worst, scale, e_psi, where)


class _BatchRealityAccumulator:
    """Keep the original whole-leaf dynamic phase check across GL chunks.

    Statistics are over raw contracted couplings, before propagators or
    Jacobians, separately for every surviving component assignment. No node
    arrays or callable results survive the current chunk.
    """

    def __init__(self):
        self._stats: dict[tuple, tuple[float, float, int, str]] = {}

    def project(self, values, e_psi, where, key):
        z = np.asarray(values) * (1j ** int(e_psi))
        re_ = np.real(z)
        im_ = np.imag(z)
        scale, worst = _batch_reality_sizes(re_, im_)
        old_scale, old_worst, _, _ = self._stats.get(key, (0.0, 0.0, 0, ""))
        # np.maximum preserves NaN as a full-batch np.max would.
        self._stats[key] = (
            float(np.maximum(old_scale, scale)),
            float(np.maximum(old_worst, worst)),
            int(e_psi),
            where,
        )
        return re_

    def validate(self):
        for scale, worst, e_psi, where in self._stats.values():
            if not _residue_negligible(worst, scale):
                raise _batch_reality_error(worst, scale, e_psi, where)


_BATCH_REALITY_PROJECT = _BatchRealityAccumulator.project


#: Default maximum Gauss-Legendre nodes per batch.  A grid of at most this
#: many nodes is the eager grid; larger grids are evaluated in blocks, which
#: bounds their sample-batch workspace at no measured cost in wall time.
DEFAULT_GL_CHUNK_SIZE = 4096


def _validate_gl_chunk_size(value):
    """Validate the optional GL batch bound without silently coercing it."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("gl_chunk_size must be a positive integer or None.")
    if value <= 0:
        raise ValueError("gl_chunk_size must be a positive integer or None.")
    return int(value)


def _validate_qmc_coupling_chunk_size(value):
    """Validate the explicit dynamic-coupling batch bound without coercion."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("qmc_coupling_chunk_size must be a positive integer or None.")
    if value <= 0:
        raise ValueError("qmc_coupling_chunk_size must be a positive integer or None.")
    return int(value)


def _qmc_couplings_in_chunks(promise, label_t, label_x, n_samples, chunk_size):
    """Assemble contracted coupling rows without partitioning R/C or reality.

    Explicit opt-in requires every raw symbol callback to be sample-separable
    with stable dtype and trailing shape. Checking the contracted output here
    cannot establish that contract for arbitrary user callbacks.
    """
    couplings = None
    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        block = np.asarray(
            promise.evaluate_at_batch(
                label_t={lab: times[start:stop] for lab, times in label_t.items()},
                label_x=label_x,
                n_samples=stop - start,
            )
        )
        if block.ndim == 0 or block.shape[0] != stop - start:
            raise ValueError(
                "qmc_coupling_chunk_size: a coupling batch must have the "
                "requested number of rows as its leading dimension."
            )
        if block.dtype.kind not in "biufc":
            raise TypeError(
                "qmc_coupling_chunk_size: coupling batches must be numeric."
            )
        if couplings is None:
            couplings = np.empty((n_samples,) + block.shape[1:], dtype=block.dtype)
        elif block.shape[1:] != couplings.shape[1:]:
            raise ValueError("qmc_coupling_chunk_size: coupling batch shape changed.")
        elif block.dtype != couplings.dtype:
            raise TypeError("qmc_coupling_chunk_size: coupling batch dtype changed.")
        couplings[start:stop] = block
    return couplings


@lru_cache(maxsize=32)
def _gauss_legendre_unit_rule(n_gauss):
    """Read-only, model-independent 1-D GL rules; never cache tensor grids."""
    from numpy.polynomial.legendre import leggauss

    nodes, weights = leggauss(n_gauss)
    u, w = (nodes + 1) / 2, weights / 2
    u.setflags(write=False)
    w.setflags(write=False)
    return u, w


def _gl_tensor_blocks(nodes, weights, n_dimensions, chunk_size):
    """Yield IJ/C-order tensor nodes and weights using bounded workspace.

    Local counters plus mixed-radix offsets avoid a full Cartesian mesh.
    Weight columns are reduced in the same axis order as the eager rule.
    Python offsets keep even very large theoretical grids lazy.
    """
    chunk_size = _validate_gl_chunk_size(chunk_size)
    if chunk_size is None:
        mesh = np.meshgrid(*([nodes] * n_dimensions), indexing="ij")
        u = np.stack([m.ravel() for m in mesh], axis=-1)
        w_mesh = np.meshgrid(*([weights] * n_dimensions), indexing="ij")
        w = np.prod(np.stack([m.ravel() for m in w_mesh], axis=-1), axis=1)
        yield u, w
        return
    n = len(nodes)
    n_points = n**n_dimensions
    for start in range(0, n_points, chunk_size):
        size = min(chunk_size, n_points - start)
        u = np.empty((size, n_dimensions))
        w_columns = np.empty_like(u)
        carry = np.arange(size, dtype=np.intp)
        offset = start
        for axis in reversed(range(n_dimensions)):
            offset, digit = divmod(offset, n)
            shifted = carry + digit
            indices = shifted % n
            carry = shifted // n
            u[:, axis] = nodes[indices]
            w_columns[:, axis] = weights[indices]
        yield u, np.prod(w_columns, axis=1)


def _sum_gl_block_totals(totals: Iterable[float]) -> float:
    """Add GL block totals, agreeing with the eager grid's ``np.sum``.

    Finite totals go to ``math.fsum``, which rounds their sum once.
    Non-finite totals are combined by floating-point addition, which gives
    what the eager sum over the nodes gives: NaN when one is NaN or both
    infinities occur, otherwise the one infinity.  fsum would raise
    ``ValueError`` for ``inf`` and ``-inf`` together.  The weights sum to
    one, so the exact sum of the finite totals cannot exceed the largest
    node value in magnitude and fsum can overflow only within rounding of
    the float range; the floating-point sum over all blocks is returned
    then.  An exception raised while a block is evaluated, an
    ``OverflowError`` from a user callable included, propagates as it
    does from the eager grid, except a ``StopIteration``, which the
    generator over the blocks turns into ``RuntimeError`` (PEP 479).
    """
    finite_sum = 0.0
    special = None

    def finite_totals():
        nonlocal finite_sum, special
        for total in totals:
            if isfinite(total):
                finite_sum += total
                yield total
            else:
                special = total if special is None else special + total

    stream = finite_totals()
    try:
        exact = fsum(stream)
    except OverflowError:
        # fsum detects its own overflow while adding the total just yielded,
        # so the stream is still suspended at that yield.  An OverflowError
        # raised while a block was evaluated has finished the stream instead,
        # and belongs to the caller.
        if inspect.getgeneratorstate(stream) != inspect.GEN_SUSPENDED:
            raise
        for _ in stream:
            pass
        exact = finite_sum
    return exact if special is None else special


def _own_copy(value: Any) -> Any:
    """``value``, or a copy of it when it is an array or a list.

    ``ndarray.copy`` keeps the dtype and the ndarray subclass; immutable
    values (Python and NumPy scalars, tuples) are returned as they are.
    """
    if isinstance(value, (np.ndarray, list)):
        return value.copy()
    return value


def _read_only_times(times: np.ndarray) -> np.ndarray:
    """``times`` with its writeable flag cleared, returned for chaining.

    The batched integrators hand the cache's batch methods views of the
    time batches they read again for the next factor: columns of one
    ``(n_samples, n_vars)`` array, and fixed-time arrays shared by every
    point pinned at one time.  A user-written cache that formed the lag in
    place (``t1 -= t2``) would change every later factor of the sample;
    with the flag cleared the write raises ``ValueError``.  Nothing is
    copied, and the package's own batch methods only read these arguments.

    The arrays the package builds for one lookup and does not read again
    are read-only as well (``PropagatorCache.C_value``, the batched
    zero-dimensional branch), so such a write raises on every path rather
    than on some.  An array that belongs to the caller gets
    :func:`_read_only_view` instead, which leaves the caller's flag alone.
    """
    times.setflags(write=False)
    return times


def _read_only_view(times: Any) -> Any:
    """``times`` when it is already read-only (or not an ndarray), else a
    read-only view of it.

    The package's own batch methods (``C_at_batch``, ``C_diagonal_batch``,
    ``C_matrix_batch``) pass their time arguments on to methods a subclass
    can override: another batch method, ``_lazy_lookup``,
    ``_closed_form_at_batch_diag``, ``_on_diagonal``.  The legacy-table
    lookups behind ``C_value`` and ``C_diagonal`` pass theirs to
    ``_on_diagonal``.  Those arguments may be the caller's own arrays, so
    they get a view whose flag is cleared: a write raises ``ValueError``,
    as on every other path, and the caller's array keeps its own flag.
    Nothing is copied.
    """
    if isinstance(times, np.ndarray) and times.flags.writeable:
        view = times.view()
        view.setflags(write=False)
        return view
    return times


def _kappa2_at(model: "PropagatorModel", n1: Any, l1: float, n2: Any, l2: float):
    """``model.kappa2(n1, l1, n2, l2)`` on copies of the two positions.

    A quadrature hands every node of one C value the same two position
    objects, and the probes hand κ² the caller's positions.  A κ² that
    writes into its arguments (``n1 -= n2``) would otherwise change every
    later node with no error, as :func:`_C_value_direct_on_copies`
    prevents for C hooks.  Scalars pass unchanged (:func:`_own_copy`).
    """
    return model.kappa2(_own_copy(n1), l1, _own_copy(n2), l2)


def _sigma2_at(model: "PropagatorModel", n1: Any, t: float, n2: Any):
    """``model.sigma2(n1, t, n2)`` on copies of the two positions; see
    :func:`_kappa2_at`."""
    return model.sigma2(_own_copy(n1), t, _own_copy(n2))


def _C_value_direct_on_copies(
    cache: "PropagatorCache", n1: Any, t1: Any, n2: Any, t2: Any, **kwargs: Any
) -> np.ndarray:
    """``cache._C_value_direct`` on copies of its array and list arguments.

    ``_C_value_direct`` may run user code: ``c_value_fn``, the closed
    form of ``Propagators.build(c_closed_form=...)``, or an override.
    Every lookup that reaches it goes through here, so that code may
    write into its position and time arguments.  Without the copies it
    would receive the caller's position arrays, a table builder's
    position reused for every cell, or an integrator's time batch read
    again later, and a write would change later values with no error.
    The two positions are copied separately, so they are distinct
    objects even when the caller passed one object for both.  Values
    are unchanged.

    A module function, not a method: a subclass overriding a copying
    method would receive the caller's objects again (an override that
    formed ``n1 -= n2`` and then called the base method took the order-2
    value to zero).  A subclass changes ``_C_value_direct``, which always
    receives the copies.
    """
    return cache._C_value_direct(
        _own_copy(n1), _own_copy(t1), _own_copy(n2), _own_copy(t2), **kwargs
    )


def _qmc_std_error(values: np.ndarray) -> float:
    """Standard error of the mean of ``values``, from 8 sub-batches, or one
    per sample below 8.  NaN for a single sample, which has no spread."""
    n_batches = min(8, len(values))
    size = len(values) // n_batches
    means = [np.mean(values[i * size : (i + 1) * size]) for i in range(n_batches)]
    return float(np.std(means, ddof=1) / np.sqrt(n_batches))


def _integration_t_min(cache: Any, t_min: float | None) -> float:
    """The lower limit of every time integral: ``t_min`` of ``cache.model``,
    from which C was built, or 0 for a cache without one.

    Raises:
        ValueError: if an explicit ``t_min`` differs from the model's.
    """
    model_t_min = getattr(getattr(cache, "model", None), "t_min", None)
    if t_min is None:
        return 0.0 if model_t_min is None else float(model_t_min)
    if model_t_min is not None and float(t_min) != float(model_t_min):
        raise ValueError(
            f"t_min={t_min} differs from the propagator model's "
            f"t_min={model_t_min}, from which C was built."
        )
    return t_min


def _table_slack(lo: float, hi: float) -> float:
    """How far past ``[lo, hi]`` a value still counts as at the end."""
    return _TABLE_EDGE_RTOL * max(abs(lo), abs(hi))


def _outside_table(values: Any, lo: float, hi: float) -> Any:
    """Where ``values`` lie outside ``[lo, hi]`` by more than rounding.

    A NaN is not outside: it stays a NaN, which the integrators refuse.
    """
    lo, hi = float(lo), float(hi)
    slack = _table_slack(lo, hi)
    values = np.asarray(values, dtype=float)
    return (values < lo - slack) | (values > hi + slack)


def _refuse_past_table(
    name: str, values: Any, lo: float, hi: float, remedy: str
) -> None:
    """Raise if any of ``values`` is outside a table's ``[lo, hi]``.

    Past its limits a lazy table clamps to the edge and a full-grid table
    extrapolates, so either would return a plausible number.

    Raises:
        ValueError: naming the first such value.
    """
    outside = _outside_table(values, lo, hi)
    if np.any(outside):
        first = np.asarray(values, dtype=float)[outside].flat[0]
        raise ValueError(
            f"C lookup at {name}={float(first)} is outside the propagator "
            f"table, which covers [{float(lo)}, {float(hi)}]; the table would "
            f"return a clamped or extrapolated value.  {remedy}"
        )


def _separation(x1: Any, x2: Any) -> float:
    """``|x1 - x2|`` of two scalar or vector positions."""
    diff = np.asarray(x1, dtype=float) - np.asarray(x2, dtype=float)
    return float(abs(diff)) if diff.ndim == 0 else float(np.linalg.norm(diff))


def _lazy_key_value(value: float) -> float:
    """``value`` to :data:`LAZY_KEY_DIGITS` significant digits."""
    return float(f"{value:.{LAZY_KEY_DIGITS - 1}e}")


def _lazy_key_separation(x1: Any, x2: Any) -> float:
    """``|x1 - x2|`` as a key, by :func:`_lazy_key_value`.  It has no floor
    tied to the size of the positions, which would move with the origin."""
    return _lazy_key_value(_separation(x1, x2))


def _separation_batch(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """``|x1 - x2|`` per sample, of ``(n,)`` scalars or ``(n, d)`` vectors."""
    diff = x1 - x2
    return np.abs(diff) if diff.ndim == 1 else np.linalg.norm(diff, axis=-1)


def _positions_per_sample(x: Any, n: int) -> np.ndarray:
    """A position as ``(n,)`` scalars or ``(n, d)`` vectors.

    ``x`` is a scalar, ``(n,)`` (one scalar per sample), ``(d,)`` with
    ``d != n`` (one fixed vector) or ``(n, d)``; see
    :meth:`PropagatorCache.C_at_batch`.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    if arr.ndim == 1 and arr.shape[0] != n:
        return np.tile(arr[None, :], (n, 1))
    return arr


def _full_grid_limits(cache: Any) -> tuple[float | None, float | None]:
    """``(r_max, x_max)`` of a cache's full spatial grid.

    ``None`` for a limit the cache's lookups do not have: a lazy table builds
    at whatever position it is asked for, cos is in ``[-1, 1]`` by
    construction, and ``c_closed_form_only`` reads no table.
    """
    if getattr(cache, "_closed_form_only", False):
        return None, None
    r_max = x_max = None
    r_range = getattr(cache, "_c_translation_r_range", None)
    if getattr(cache, "_c_translation_splines", None) is not None and r_range:
        r_max = float(r_range[1])
    axes = getattr(cache, "_c_general_axes", None)
    if getattr(cache, "_c_general_interpolators", None) is not None and axes:
        x_max = float(axes[2][-1])
    return r_max, x_max


def _unit_vector(x: Any) -> np.ndarray:
    """``x / |x|`` as a 1-D array; the zero vector stays zero."""
    v = np.atleast_1d(np.asarray(x, dtype=float))
    norm = float(np.linalg.norm(v))
    return v if norm == 0.0 else v / norm


def _rotation_cos(x1, x2) -> float:
    """Cosine similarity ``x1·x2 / (|x1| |x2|)`` as a scalar.

    Works for scalar x (interpreted as 1-D vector) and array x.
    Returns ``1.0`` when either vector is exactly zero (degenerate;
    avoids a divide-by-zero NaN).
    """
    v1 = np.atleast_1d(np.asarray(x1, dtype=float))
    v2 = np.atleast_1d(np.asarray(x2, dtype=float))
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        return 1.0
    return float(np.dot(v1, v2) / (n1 * n2))


def _resolve_integrate_over(
    integrate_over: Any,
    ext_vars: list[str],
) -> set[str]:
    """Normalise the ``integrate_over`` kwarg into a set of external
    point names to integrate over.

    Accepts:

    - ``None``: the empty set (all externals fixed at ``lambda_f``).
      This is the physics-observable convention
      ``⟨φ(t_f) · φ(t_f)⟩``.
    - ``"all"``: the full set ``ext_vars`` (time-integrated moment).
    - Iterable of names: used as-is, after validating that each
      name appears in ``ext_vars``.

    Raises ``ValueError`` on an unknown external name.
    """
    if integrate_over is None:
        return set()
    if isinstance(integrate_over, str):
        if integrate_over == "all":
            return set(ext_vars)
        raise ValueError(
            f"integrate_over must be None, 'all', or an iterable of "
            f"external-point names; got string {integrate_over!r}."
        )
    requested = set(integrate_over)
    unknown = requested - set(ext_vars)
    if unknown:
        raise ValueError(
            f"integrate_over references external points that don't "
            f"exist in this diagram: {sorted(unknown)}.  Known "
            f"externals: {ext_vars}."
        )
    return requested


def _unique_time_pairs(tl: np.ndarray, tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The distinct rows of the time pairs ``(tl[s], tr[s])``, shape
    ``(m, 2)``, and the index of each pair's row, shape ``(n,)``."""
    pairs = np.stack([tl, tr], axis=1)
    uniq, inverse = np.unique(pairs, axis=0, return_inverse=True)
    return uniq, np.asarray(inverse).reshape(-1)


class PropagatorCache:
    """Evaluates and caches propagator values.

    Computes C via double integration of ``R · κ · R`` and caches results
    to avoid redundant quadrature.

    Args:
        model: The physical propagator model.
        quad_opts: Options passed to ``scipy.integrate.dblquad``, over
            :data:`_QUAD_TOLERANCES` (e.g. ``{'epsrel': 1e-10}``).
    """

    def __init__(
        self,
        model: PropagatorModel,
        quad_opts: dict | None = None,
        homogeneity: str = "translation",
        interp_method: str = "linear",
        c_method: str = "auto",
        n_gauss: int = DEFAULT_C_N_GAUSS,
        c_value_fn: Callable | None = None,
    ):
        """Initialise the propagator cache.

        Args:
            model: Propagator model specifying R and κ².
            quad_opts: Options forwarded to ``scipy.integrate.dblquad``, over
                :data:`_QUAD_TOLERANCES`.  An entry of C that cancels to 0
                cannot meet a relative tolerance, so scipy warns on it unless
                ``epsabs`` is given.
            c_value_fn: Optional ``(n1, t1, n2, t2) -> (N, N)`` callable that
                **replaces** the quadrature construction of C entirely.

                By default C is *derived* from R and the noise cumulant as
                ``C = ∫∫ R κ R``.  That relation does not hold for every
                physically meaningful propagator pair, notably a
                disorder-averaged (DMFT) solution, where
                ``⟨R κ R⟩ ≠ ⟨R⟩ κ ⟨R⟩``, so such a C must be supplied
                directly.  This is the public, L0 equivalent of the L1
                ``Propagators.build(c_closed_form=..., c_closed_form_only=True)``
                route; previously it required subclassing and overriding
                :meth:`_C_value_direct`.

                It is consulted by :meth:`_C_value_direct`, hence by
                :meth:`C_value`, :meth:`C_diagonal` and every
                ``precompute_C_table_*`` builder.  Array and list
                arguments reach it as copies, so it may write into them
                (:func:`_C_value_direct_on_copies`).  The copies are
                shallow: the entries of a nested list or an object array
                are the caller's.
            c_method: Quadrature method used by :meth:`_C_value_direct`
                (and therefore by every ``precompute_C_table_*`` builder).

                - ``'auto'`` (default): each pair of positions has its
                  own rule (:meth:`_c_rule`): Gauss-Legendre at the
                  count :func:`select_gl_node_count` finds, which
                  refines the rule until it agrees with itself at the
                  probe cells up to the horizon.  When no node count
                  up to the cap converges it takes the graded rule of
                  :func:`select_graded_levels`, then ``'dblquad'`` with
                  a warning for that pair.  A direct :meth:`C_value`
                  call takes the rule of its pair at the table's
                  horizon, or, before any table and past it, at a
                  horizon worked out from the cell.
                - ``'gauss_legendre'``: tensor-product Gauss-Legendre
                  with diagonal-aware sub-region splitting at
                  ``λ1 = λ2`` and exactly ``n_gauss`` nodes, with no
                  convergence check.  ~10-1000× faster than
                  ``'dblquad'`` on smooth κ² with a single ``|λ1−λ2|``
                  cusp on the diagonal (the demo1/demo2 OU-style
                  kernels).  See :func:`_C_value_direct_gl`.  Its
                  accuracy at fixed ``n_gauss`` degrades as
                  ``(γ + 1/σ_t) · t_max`` grows (measured: 1e-12 at
                  ``t_max = 15``, 1e-5 at 30, 2e-2 at 100 for the
                  demo1 kernel with ``n_gauss=20``), which is why
                  ``'auto'`` re-checks instead of trusting 20.
                - ``'dblquad'``: adaptive 2-D quadrature via
                  :func:`scipy.integrate.dblquad`, 10-250 ms per call,
                  with no convergence check.  It stops when its first
                  samples agree, so a kernel far narrower than the
                  interval can come back wrong by any factor with a small
                  error estimate.

                Each ``precompute_C_table_*`` builder also accepts a
                local ``c_method`` kwarg that overrides this default
                for the duration of the build.
            n_gauss: Number of Gauss-Legendre nodes per dimension per
                sub-region for ``c_method='gauss_legendre'``, and the
                STARTING node count for ``'auto'``.  Default ``20``.
                Cost scales as ``n_gauss²`` per sub-region (3
                sub-regions for ``t1 ≠ t2``, 2 for ``t1 = t2``).
            homogeneity: Physical assumption about how C depends on the
                spatial coordinates.  Determines which
                ``precompute_C_table_*`` builder is valid.

                - ``'translation'`` (default): C depends only on
                  ``|x1 − x2|``.  Appropriate for translation-invariant
                  (spatially stationary) noise, the most common case.
                  Build via :meth:`precompute_C_table_translation`.
                - ``'rotation'``: C depends only on ``x1 · x2``.
                  Appropriate when x is a unit direction vector (e.g.
                  on the sphere).  Build via
                  :meth:`precompute_C_table_rotation`.
                - ``'general'``: no symmetry assumed; C is evaluated on
                  a full 4-D grid in ``(t1, t2, x1, x2)``.  Build via
                  :meth:`precompute_C_table_general`.

                When a spatial builder is called with its ``*_grid``
                argument left ``None``, the cache enters **lazy mode**
                for that symmetry: 2-D ``(t1, t2)`` splines are built
                on-demand for each distinct parameter value queried,
                and memoized.  This is the right default for
                moment-at-fixed-points workflows where only a few
                distinct parameter values ever appear.
            interp_method: ``RegularGridInterpolator`` method used by
                the three full-grid spline builders
                (``precompute_C_table_translation``,
                ``precompute_C_table_rotation``,
                ``precompute_C_table_general``).

                - ``'linear'`` (default, safe): O(h²) accuracy,
                  monotone, never overshooting or flipping sign even on
                  steeply-decaying C tails. Required when C spans
                  many orders of magnitude across the r-grid (the
                  typical cosmological setting; see
                  ``tests/test_evaluate_interpolation_accuracy.py``).
                - ``'cubic'``: O(h⁴) accuracy on smooth, well-sampled
                  grids; only safe when the user knows their C is
                  bounded away from grid-induced sign-flip artefacts.

                ``RegularGridInterpolator`` accepts any other method
                name supported by the installed scipy (e.g.
                ``'quintic'``, ``'pchip'``, ``'nearest'``); they are
                forwarded as-is and validated lazily on table build.
        """
        if homogeneity not in HOMOGENEITIES:
            raise ValueError(
                f"homogeneity must be 'translation', 'rotation', or "
                f"'general'; got {homogeneity!r}"
            )
        if c_method not in C_METHODS:
            raise ValueError(
                f"c_method must be 'auto', 'dblquad' or 'gauss_legendre'; "
                f"got {c_method!r}"
            )
        if int(n_gauss) < 2:
            raise ValueError(f"n_gauss must be >= 2; got {n_gauss!r}")
        self.model = model
        self.quad_opts = {**_QUAD_TOLERANCES, **(quad_opts or {})}
        self.homogeneity = homogeneity
        self.interp_method = interp_method
        self.c_method = c_method
        self.n_gauss = int(n_gauss)
        #: Rules of :meth:`_c_rule`, by pair of positions and horizon.
        self._c_rules: dict = {}
        #: The error that ended a search of those that ended in one, as text.
        self._c_rule_raised: dict = {}
        #: What has been warned about, so each warning comes once.
        self._c_rule_warnings: set = set()
        #: The rules of the last table: pair of positions to rule under
        #: ``'auto'``, the one rule under the key ``None`` otherwise.
        self._table_rules: dict = {}
        #: The ``t_max`` of that table when ``'auto'`` chooses its rules.
        self._table_horizon: float | None = None
        #: Optional user-supplied C, bypassing the ``∫∫ R κ R`` construction.
        self.c_value_fn = c_value_fn

        # Closed-form-only path: skip every spline and route C lookups
        # directly through ``self._C_value_direct``. Set by
        # ``Propagators.build()`` when both ``c_closed_form`` and
        # ``c_closed_form_only=True`` are passed; the user's analytical
        # C function then becomes the lookup function with no
        # interpolation error. ``closed_form_vectorized`` flags whether
        # the user's c_fn accepts batched (n,)-shape t and (n,)- or
        # broadcast-scalar n inputs and returns ``(n, N, N)``; if not,
        # the per-sample fallback is used (slow, but correct).
        self._closed_form_only: bool = False
        self._closed_form_vectorized: bool = False

        #: Bounded LRU memo for ``C_value`` (``move_to_end`` on every hit,
        #: eviction from the cold end).  It was an UNBOUNDED plain dict
        #: described in comments as an LRU, so on a long sweep it grew without
        #: limit.  The bound is on entry COUNT: one entry is an
        #: ``(N, N)`` correlator, so the memory ceiling scales as N^2 and a
        #: large-N run should lower :data:`_C_CACHE_MAXSIZE` or call
        #: :meth:`clear_cache`.  Excluded from pickling; see
        #: :meth:`__getstate__`.
        self._c_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        # Legacy 2-D (t1, t2) spline at fixed x, still populated by
        # ``precompute_C_table`` for backward compatibility.
        self._c_splines: list | None = None
        #: 1-D splines of C(t, t) harvested from the same grid; see
        #: ``precompute_C_table``.  The 2-D tensor-product spline is C^2 by
        #: construction and cannot represent the diagonal kink.
        self._c_diag_splines: list | None = None
        self._c_table_range: tuple[float, float] | None = None

        # Full N-D spatial splines, populated by precompute_C_table_*
        # with an explicit parameter grid:
        self._c_translation_splines: list | None = None  # 3-D (t1,t2,r)
        self._c_translation_r_range: tuple[float, float] | None = None
        self._c_rotation_splines: list | None = None  # 3-D (t1,t2,cos)
        self._c_general_interpolators: list | None = None  # 4-D
        self._c_general_axes: tuple | None = None

        # Lazy per-parameter-value 2-D (t1, t2) spline caches.  A
        # :class:`_LazyTimeSplineCache` is attached when the user
        # calls a ``precompute_C_table_*`` method without a parameter
        # grid (see ``lazy mode`` in the docstring).  Keyed by
        # rounded parameter value (r, cos, or (x1, x2)).
        self._lazy_translation: _LazyTimeSplineCache | None = None
        self._lazy_rotation: _LazyTimeSplineCache | None = None
        self._lazy_general: _LazyTimeSplineCache | None = None

    def R_time(self, t_left: float, t_right: float) -> float | np.ndarray:
        """Evaluate R_time(t_left, t_right), the RAW model accessor.

        Returns scalar if ``model.iso_R=True``, else ``(N, N)`` array.

        Θ is **not** applied here.  This is the single authoritative
        statement of the convention, which used to be described three
        inconsistent ways across this module:

        * ``R_time`` is Θ-stripped, so a model author writes only the
          retarded branch and never has to encode the step function.
        * Θ is applied at **diagram evaluation**, by every consumer that
          multiplies R factors together: :meth:`R_product`,
          ``DiagramIntegrand._evaluate_r_product_general``, and the batched
          :meth:`R_time_batch` and :meth:`R_matrix_batch`, which share
          :meth:`_causal_R`.  Each asks :func:`_retarded`, and none calls
          this method for an acausal pair.
        * The older claim that "the integration domain handles causality"
          holds only where an integration domain exists.  An R propagator
          joining two *fixed external* points has none; see
          :meth:`R_product`.

        Callers wanting the physical, Θ-enforced propagator should use
        :meth:`R_time_batch` (or evaluate a diagram), not this method.
        """
        return self.model.R_time(t_left, t_right)

    def R_product(
        self,
        r_pairs: tuple[tuple[str, str], ...],
        times: dict[str, float],
    ) -> float:
        """Product of R_time over all R propagator pairs.

        Only valid when ``model.iso_R=True`` (R is scalar).

        Retardation is enforced here.  ``R_time`` itself is the raw model
        accessor and does not apply Θ ("the integration domain
        handles causality"), but that only holds when there *is* an
        integration domain.  An R propagator joining two **fixed external**
        points has none: at order 0, ``<phi(t_x) psi(t_y)>`` would otherwise
        evaluate to the unbounded acausal ``exp(+mu (t_y - t_x))`` for
        ``t_y > t_x``.  Under the Itô prescription equal times give 0 as well.
        """
        result = 1.0
        for sl, sr in r_pairs:
            t_l, t_r = times[sl], times[sr]
            if not _retarded(t_l, t_r):
                return 0.0
            result *= float(self.R_time(t_l, t_r))
        return result

    def C_value(
        self,
        n1: Any,
        t1: float,
        n2: Any,
        t2: float,
    ) -> np.ndarray:
        """Compute the full C matrix ``C_{ab}(n1, t1; n2, t2)``.

        If :meth:`precompute_C_table` has been called and ``t1``, ``t2``
        are within the table range, uses fast spline interpolation.
        Otherwise evaluates the integral by :meth:`_C_value_direct`::

            C_{ab} = ∫_{t_min}^{t1} dλ' ∫_{t_min}^{t2} dλ''
                     R_time(t1,λ') κ_{ab}(n1,λ'; n2,λ'') R_time(t2,λ'')

        A spatial table takes precedence and has no such fallback: a point
        outside it is refused, as in :meth:`C_at_batch`.

        Returns:
            ``(N, N)`` array.
        """
        # Fast path (spatial-aware): route through C_at_batch which
        # dispatches on ``homogeneity`` + full/lazy table presence.
        # Only used when a spatial table has actually been built;
        # otherwise fall through to the legacy/direct paths so the
        # plain ``PropagatorCache(model)`` + legacy ``precompute_C_table``
        # flow stays bit-identical to before.
        if _built_spatial_table(self):
            N = self.model.n_components
            # Read-only, as every time array passed to a batch method is
            # (see _read_only_times): an override of C_at_batch that writes
            # into its time arguments raises here as on the integrators.
            t1_arr = _read_only_times(np.array([t1], dtype=float))
            t2_arr = _read_only_times(np.array([t2], dtype=float))
            # n1, n2 may be scalars or arrays (e.g. np.float64 or
            # small np.ndarray); converted so the downstream
            # rotation/general paths see consistent shape.  Copies, not
            # views: the caller reads its positions again, and an override
            # of C_at_batch may write into its position arguments.
            x1_arr = (
                np.array(n1, dtype=float).reshape(1, -1)
                if np.ndim(n1) > 0
                else np.array([float(n1)])
            )
            x2_arr = (
                np.array(n2, dtype=float).reshape(1, -1)
                if np.ndim(n2) > 0
                else np.array([float(n2)])
            )
            if x1_arr.ndim == 2 and x1_arr.shape[-1] == 1:
                x1_arr = x1_arr.ravel()
                x2_arr = x2_arr.ravel()
            c_tab = self.C_at_batch(t1_arr, t2_arr, x1_arr, x2_arr)[0]
            if c_tab.ndim == 2:
                # Full table (diag_C=False): every C_ab is stored.
                return np.array(c_tab, dtype=float)
            C_mat = np.zeros((N, N))
            for a in range(N):
                C_mat[a, a] = c_tab[a]
            return C_mat

        # Fast path (legacy time-only): 2-D spline table
        if self._c_splines is not None and self._legacy_table_serves_model():
            lo, hi = self._c_table_range  # type: ignore[misc]
            if lo <= t1 <= hi and lo <= t2 <= hi:
                return _owned_lookup(
                    self,
                    PropagatorCache._C_value_from_table,
                    self._C_value_from_table(t1, t2),
                )

        # Check the (n1, t1, n2, t2) memo
        # Key on CONTENTS, not id().  `id()` is unique only among LIVE
        # objects: CPython recycles the address of a freed temporary, so two
        # different position arrays could collide and the memo then returned
        # one separation's C for another.
        #
        # A fixed-size digest, rather than the raw bytes: the digest is 16
        # bytes whatever the array's size, so there is no size cutoff, no
        # escape-hatch sentinel, and no "is this key cacheable" question to
        # get wrong.  An earlier version answered it with a membership test
        # that saw only the key's top level, so an oversized array nested in a
        # list left the sentinel buried inside and two different positions
        # shared a key.  Removing the special case removes that whole class.
        # ``tobytes()`` already serialises in C order whatever the array's
        # memory layout, so it normalises strides on its own.  Do NOT route it
        # through ``np.ascontiguousarray`` first: that downcasts an ndarray
        # SUBCLASS to a base array, discarding the subclass's own ``tobytes``
        # (a masked array then keys on its raw buffer, and two positions
        # differing only in their mask share one entry).
        cacheable = True

        def _cache_key_part(obj):
            nonlocal cacheable
            if isinstance(obj, np.ndarray):
                if obj.dtype.kind == "O":
                    # An object array's buffer is raw PyObject POINTERS, and
                    # an address is unique only among LIVE objects.  CPython
                    # recycles a freed temporary's address, so two different
                    # positions would collide.  Refuse rather than guess.
                    cacheable = False
                    return None
                return (
                    type(obj).__name__,
                    obj.shape,
                    obj.dtype.str,
                    hashlib.blake2b(obj.tobytes(), digest_size=16).digest(),
                )
            if isinstance(obj, (list, tuple)):
                return tuple(_cache_key_part(v) for v in obj)
            return obj

        cache_key = (_cache_key_part(n1), t1, _cache_key_part(n2), t2)
        # ``cacheable`` is set from INSIDE the recursion, so an object array
        # nested in a list is caught too.  An earlier version tested only the
        # key's top level and let nested cases share an entry.
        if cacheable:
            hit = self._c_cache.get(cache_key)
            if hit is not None:
                return hit

        # Delegate to ``_C_value_direct`` (which adds the white-noise
        # 1-D contribution when ``model.sigma2`` is set), then store
        # in the per-arg memo.
        # COPY on both branches.  ``_C_value_direct`` ends in
        # ``np.asarray(...)``, which is the identity for a float64 array, so
        # without the copy the result is the C hook's own object.  A
        # ``c_value_fn`` that fills and returns one preallocated buffer would
        # then change an earlier result at the next lookup, on the branch that
        # skips the memo (object-dtype positions) too, and freezing the result
        # would flip the flag on the user's buffer or on a module-level
        # constant correlator.
        C_mat = np.array(_C_value_direct_on_copies(self, n1, t1, n2, t2), copy=True)
        if self.model.diag_C and C_mat.ndim == 2:
            # A user C may hold C_ab; the batched reads and the tables keep
            # C_aa only, so this read does too.
            C_mat = np.diag(np.diag(C_mat))
        if cacheable:
            # Handed out read-only: the array returned IS the memo entry, so a
            # caller doing ``C += x`` would silently rewrite every later
            # lookup.  Failing loudly beats corrupting the cache.
            C_mat.flags.writeable = False
            self._c_cache[cache_key] = C_mat
            if len(self._c_cache) > _C_CACHE_MAXSIZE:
                self._c_cache.popitem(last=False)
        return C_mat

    def C_diagonal(
        self,
        n: Any,
        t1: float,
        n_prime: Any | None = None,
        t2: float | None = None,
    ) -> np.ndarray:
        """Return diagonal ``[C_{00}, C_{11}, …]`` as a 1D array.

        If ``n_prime`` and ``t2`` are None, uses equal-point ``C(n,t; n,t)``.
        Uses spline table when available for fast lookup.
        """
        if n_prime is None:
            n_prime = n
        if t2 is None:
            t2 = t1
        # Fast path: legacy time-only spline table, but ONLY when there is
        # no spatial table.  `C_value` checks the spatial table FIRST, so
        # taking the legacy one here regardless meant the two accessors
        # disagreed by ~38% whenever both tables were present: this one is
        # position-blind, that one is not.  Same precedence in both now.
        if (
            self._c_splines is not None
            and self.model.diag_C
            and not _cache_has_spatial_table(self)
        ):
            lo, hi = self._c_table_range  # type: ignore[misc]
            if lo <= t1 <= hi and lo <= t2 <= hi:
                return _owned_lookup(
                    self,
                    PropagatorCache._C_diagonal_from_table,
                    self._C_diagonal_from_table(t1, t2),
                )
        # Copies for an overridden C_value (see _c_lookup_positions).
        n, n_prime = _c_lookup_positions(self, "C_value", n, n_prime)
        C_mat = self.C_value(n, t1, n_prime, t2)
        return np.diag(C_mat)

    def __getstate__(self) -> dict:
        """Pickle without the ``C_value`` memo.

        Every parallel builder (``precompute_C_table_translation`` /
        ``_rotation`` / ``_general``) dispatches a closure that references
        ``self``, so the whole cache object is serialised into each worker
        payload.  A full memo is tens of megabytes at a modest number of
        components and grows as N^2, and a worker cannot benefit from the
        parent's entries anyway; it recomputes what it needs.
        """
        state = self.__dict__.copy()
        state["_c_cache"] = OrderedDict()
        return state

    def __setstate__(self, state: dict) -> None:
        """Refuse a cache pickled by an earlier release, whose rule state
        this one cannot read.  ``load_or_compute`` rebuilds such a file."""
        if "_c_cache" in state and "_table_horizon" not in state:
            raise ValueError(
                "this PropagatorCache was pickled by an earlier release of "
                "sft-wick, and a pickled cache does not cross releases.  "
                "Rebuild it, or use cache_path, which does."
            )
        self.__dict__.update(state)
        self.__dict__.setdefault("_c_rule_raised", {})

    def clear_cache(self) -> None:
        """Clear the C value cache and spline table, and the rules of that
        table.  A spatial table stays, with its range and rules."""
        self._c_cache.clear()
        self._c_splines = None
        self._c_splines_entries = None
        self._c_diag_splines = None
        if not _cache_has_spatial_table(self):
            self._c_table_range = None
            self._table_rules = {}
            self._table_horizon = None

    def precompute_C_table(
        self,
        t_max: float,
        n_grid: int = 100,
        direction: Any = 0,
    ) -> None:
        """Pre-compute C on a grid and build spline interpolators.

        For ``iso_R + diag_C`` with direction-independent ``kappa2``,
        ``C_{aa}(t1, t2)`` depends only on ``(t1, t2)``.  This method
        evaluates C on an ``n_grid × n_grid`` grid under ``c_method``,
        then builds a :class:`~scipy.interpolate.RectBivariateSpline`
        for each diagonal component, and for every ``C_ab`` when the model
        has ``diag_C=False``.

        After calling this, :meth:`C_value` and :meth:`C_diagonal`
        use fast spline interpolation instead of direct quadrature.

        Args:
            t_max: Upper bound of the time grid.
            n_grid: Number of grid points per axis (default 100).
            direction: Direction value to use for kappa2 evaluation.
        """
        from scipy.interpolate import RectBivariateSpline

        # A float, not the caller's object: the table keeps it as its range,
        # as in precompute_C_table_translation.
        t_max = float(t_max)
        # The grid is read through C_value, which serves a table first.
        self.clear_cache()
        m = self.model
        t_min = m.t_min
        ts = np.linspace(t_min, t_max, n_grid)
        entries = self._c_table_entries()
        self._resolve_build_method(None, None, t_max, [(direction, direction)])

        # One grid per stored entry: C_aa, or every C_ab under diag_C=False.
        grids = [np.zeros((n_grid, n_grid)) for _ in entries]

        for i in range(n_grid):
            for j in range(n_grid):
                # ``direction`` is both ends of every node: copies for an
                # overridden C_value (see _c_lookup_positions).
                x1, x2 = _c_lookup_positions(self, "C_value", direction, direction)
                C_mat = self.C_value(x1, ts[i], x2, ts[j])
                for k, (a, b) in enumerate(entries):
                    grids[k][i, j] = C_mat[a, b]

        self._c_splines = [RectBivariateSpline(ts, ts, g) for g in grids]
        self._c_splines_entries = entries
        # Harvest the i == j entries into a separate 1-D spline.  Zero extra
        # quadrature: they are already in ``grids``.
        #
        # ``C(t1,t2) = int_0^{min(t1,t2)} R(t1,l) sigma2(l) R(t2,l) dl`` has a
        # derivative discontinuity of exactly ``-sigma2(t)`` on the diagonal:
        # approaching from t1 < t2 the moving upper limit contributes an extra
        # ``R(t1,t1) sigma2(t1) R(t2,t1)``, absent from the other side.  A
        # tensor-product spline is C^2 everywhere, so it smears that kink and
        # stops converging there: measured 22.3% relative error at n_grid=41
        # and still 21.4% at n_grid=321, i.e. p = 0.009, no convergence at all,
        # while the same table is clean O(h^4) away from the diagonal.  Every
        # tadpole evaluates C(s,s), exactly on the kink.
        #
        # Along the diagonal itself ``t -> C(t,t)`` is smooth, so a 1-D cubic
        # spline restores O(h^4).
        self._c_diag_splines = [_diag_line_interp(ts, np.diag(g)) for g in grids]
        self._c_table_range = (t_min, t_max)
        self._c_table_direction = direction

    @property
    def has_table(self) -> bool:
        """Whether a pre-computed C table is available."""
        return self._c_splines is not None

    def _on_diagonal(self, t1, t2):
        """Whether ``t1`` and ``t2`` are the same time on the legacy table.

        The tolerance is a fraction of the table's span, so it does not move
        with the origin of time, or rounding of the times where that is larger.
        """
        lo, hi = self._c_table_range
        rounding = _DIAG_ROUNDING * max(abs(lo), abs(hi))
        tol = max(_DIAG_TOL * (hi - lo), rounding)
        return np.abs(np.asarray(t1) - np.asarray(t2)) <= tol

    def _legacy_entries(self) -> tuple:
        """The ``(a, b)`` pairs the legacy table stores, in storage order.

        A table pickled before the pairs were recorded holds ``C_aa``.
        """
        entries = getattr(self, "_c_splines_entries", None)
        if entries is None:
            return _c_table_entries(self.model.n_components, True)
        return entries

    def _legacy_table_serves_model(self) -> bool:
        """Whether the legacy table holds every entry :meth:`C_value`
        returns: ``C_aa`` under ``diag_C``, every ``C_ab`` otherwise."""
        if self.model.diag_C:
            return True
        return len(self._legacy_entries()) == self.model.n_components**2

    def _legacy_diag_ks(self) -> list:
        """Positions of ``C_00, C_11, …`` among the legacy table's entries."""
        where = {e: k for k, e in enumerate(self._legacy_entries())}
        return [where[(a, a)] for a in range(self.model.n_components)]

    def _C_diagonal_from_table(self, t1: float, t2: float) -> np.ndarray:
        """Look up C diagonal from spline table.

        Routes equal times through the 1-D diagonal spline: the 2-D spline
        cannot represent the kink there (see ``precompute_C_table``).
        """
        # Read-only views for _on_diagonal, which a subclass may override and
        # after which the times are read again (see _read_only_view).
        t1, t2 = _read_only_view(t1), _read_only_view(t2)
        ks = self._legacy_diag_ks()
        if self._c_diag_splines is not None and self._on_diagonal(t1, t2):
            t = 0.5 * (float(t1) + float(t2))
            return np.array(
                [float(_diag_line_eval(self._c_diag_splines[k], t)[0]) for k in ks]
            )
        # ``np.squeeze`` keeps this robust across SciPy versions: newer
        # SciPy returns a 1-element 1-D array from a scalar ``grid=False``
        # call, and ``float()`` on a non-0-D array raises under NumPy >= 2.
        return np.array(
            [
                float(np.squeeze(self._c_splines[k](t1, t2, grid=False)))  # type: ignore[index]
                for k in ks
            ]
        )

    def _C_value_from_table(self, t1: float, t2: float) -> np.ndarray:
        """Look up C matrix from spline table: every stored entry, zeros
        elsewhere (the off-diagonal of a ``diag_C`` table).

        Shares :meth:`_C_diagonal_from_table`'s diagonal-kink routing so the
        two accessors cannot disagree about C(t, t).
        """
        # Read-only views for _on_diagonal (see _C_diagonal_from_table).
        t1, t2 = _read_only_view(t1), _read_only_view(t2)
        N = self.model.n_components
        C_mat = np.zeros((N, N))
        entries = self._legacy_entries()
        if all(a == b for a, b in entries):
            diag = self._C_diagonal_from_table(t1, t2)
            for a in range(N):
                C_mat[a, a] = diag[a]
            return C_mat
        on = self._c_diag_splines is not None and self._on_diagonal(t1, t2)
        t = 0.5 * (float(t1) + float(t2))
        for k, (a, b) in enumerate(entries):
            if on:
                C_mat[a, b] = float(_diag_line_eval(self._c_diag_splines[k], t)[0])
            else:
                C_mat[a, b] = float(
                    np.squeeze(self._c_splines[k](t1, t2, grid=False))
                )  # type: ignore[index]
        return C_mat

    # --- Vectorized methods for batch evaluation ---

    def _refuse_times_past_table(self, t1: np.ndarray, t2: np.ndarray) -> None:
        """Raise if a time is outside ``_c_table_range``, if there is one."""
        if self._c_table_range is None:
            return
        lo, hi = self._c_table_range
        remedy = "Rebuild the propagators with a t_max that covers it."
        _refuse_past_table("t1", t1, lo, hi, remedy)
        _refuse_past_table("t2", t2, lo, hi, remedy)

    def C_diagonal_batch(
        self,
        t1: np.ndarray,
        t2: np.ndarray,
    ) -> np.ndarray:
        """Evaluate diagonal C at arrays of time pairs.

        Requires :meth:`precompute_C_table` to have been called.

        Args:
            t1: Array of shape ``(n,)``, the first time coordinates.
            t2: Array of shape ``(n,)``, the second time coordinates.

        Returns:
            Array of shape ``(n, N)`` where ``N`` is the number of
            field components.  ``result[s, a]`` is ``C_{aa}(t1[s], t2[s])``.

        Raises:
            ValueError: if a time is outside the table.
        """
        if self._c_splines is None:
            raise RuntimeError("C_diagonal_batch requires precompute_C_table()")
        # Read-only views for _on_diagonal, which a subclass may override
        # (see _read_only_view).
        t1 = _read_only_view(np.asarray(t1))
        t2 = _read_only_view(np.asarray(t2))
        self._refuse_times_past_table(t1, t2)
        n = len(t1)
        N = self.model.n_components
        ks = self._legacy_diag_ks()
        result = np.empty((n, N))
        for a, k in enumerate(ks):
            result[:, a] = self._c_splines[k](t1, t2, grid=False)
        if self._c_diag_splines is not None:
            # Tadpoles evaluate C(s, s): both legs are the SAME sampled time,
            # so this mask is hit exactly, not approximately.
            on_diag = self._on_diagonal(t1, t2)
            if np.any(on_diag):
                td = 0.5 * (
                    np.asarray(t1, dtype=float)[on_diag]
                    + np.asarray(t2, dtype=float)[on_diag]
                )
                for a, k in enumerate(ks):
                    result[on_diag, a] = _diag_line_eval(self._c_diag_splines[k], td)
        return result

    def C_matrix_batch(
        self,
        t1: np.ndarray,
        t2: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the full C matrix at arrays of time pairs.

        The ``(n, N, N)`` counterpart of :meth:`C_diagonal_batch`, from the
        same :meth:`precompute_C_table` table: every ``C_ab`` when it was
        built for ``diag_C=False``, the diagonal (zeros elsewhere) for a
        ``diag_C`` table.  The batched integrators call it in place of
        :meth:`C_diagonal_batch` when ``model.diag_C`` is False.
        """
        if not isinstance(self._c_splines, (list, tuple)):
            raise RuntimeError("C_matrix_batch requires precompute_C_table()")
        # Read-only views for _on_diagonal (see _read_only_view).
        t1a = _read_only_view(np.atleast_1d(np.asarray(t1, dtype=float)))
        t2a = _read_only_view(np.atleast_1d(np.asarray(t2, dtype=float)))
        self._refuse_times_past_table(t1a, t2a)
        n = len(t1a)
        N = self.model.n_components
        result = np.zeros((n, N, N))
        on_diag = (
            self._on_diagonal(t1a, t2a)
            if self._c_diag_splines is not None
            else np.zeros(n, bool)
        )
        td = 0.5 * (t1a[on_diag] + t2a[on_diag])
        for k, (a, b) in enumerate(self._legacy_entries()):
            col = np.array(self._c_splines[k](t1a, t2a, grid=False), dtype=float)
            if np.any(on_diag):
                col[on_diag] = _diag_line_eval(self._c_diag_splines[k], td)
            result[:, a, b] = col
        return result

    def R_time_batch(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        """Evaluate R_time at arrays of time pairs (iso_R only; a
        matrix-valued R goes through :meth:`R_matrix_batch`).

        Args:
            t1: Array of shape ``(n,)``, the left times.
            t2: Array of shape ``(n,)``, the right times.

        Returns:
            Array of shape ``(n,)``.  R(t1, t2) = Θ(t1−t2) × R_time(t1, t2).
        """
        return self._causal_R(t1, t2, ())

    def _causal_R(self, t1: Any, t2: Any, entry: tuple) -> np.ndarray:
        """``Θ(t1 − t2) R_time(t1, t2)`` per time pair, each value of shape
        ``entry``: ``()`` for a scalar R, ``(N, N)`` for a matrix R.

        Θ is applied here: an R propagator joining two fixed external points
        has no integration domain to enforce it (see :meth:`R_product`).

        The user's R_time is called ONLY on the causal pairs, matching
        :meth:`R_product` and ``_evaluate_r_product_general``, which
        short-circuit before calling it.  Evaluating everywhere and masking
        afterwards would give the same number, but a spurious exception or
        RuntimeWarning through this path alone for a model whose R_time
        raises or overflows on acausal input.

        A callable flagged ``vectorized`` is called once on the arrays of
        causal pairs; any other once per distinct causal pair (on a
        Gauss-Legendre grid most samples repeat a pair).
        """
        t1a, t2a = np.broadcast_arrays(
            np.asarray(t1, dtype=float), np.asarray(t2, dtype=float)
        )
        shape = t1a.shape
        t1a, t2a = t1a.reshape(-1), t2a.reshape(-1)
        out = np.zeros((t1a.shape[0], *entry))
        causal = _retarded(t1a, t2a)
        R = self.model.R_time
        if causal.any() and getattr(R, "vectorized", False):
            tl, tr = t1a[causal], t2a[causal]
            vals = np.asarray(R(tl, tr), dtype=float)
            if vals.shape != (tl.shape[0], *entry):
                kind = "a scalar R (iso_R)"
                if entry:
                    kind = "a matrix R (model.iso_R is False)"
                raise ValueError(
                    f"a vectorized R_time with {kind} must return shape "
                    f"{(tl.shape[0], *entry)} for {tl.shape[0]} time pairs; "
                    f"got shape {vals.shape}."
                )
            out[causal] = vals
        elif causal.any():
            uniq, inverse = _unique_time_pairs(t1a[causal], t2a[causal])
            vals = np.empty((uniq.shape[0], *entry))
            for k, (tl, tr) in enumerate(uniq):
                value = R(float(tl), float(tr))
                if entry:
                    value = np.asarray(value, dtype=float)
                    if value.shape != entry:
                        raise ValueError(
                            f"a matrix-valued R (model.iso_R is False) must "
                            f"return an {entry} array from R_time; got shape "
                            f"{value.shape} at t = ({tl!r}, {tr!r})."
                        )
                else:
                    value = float(value)
                vals[k] = value
            out[causal] = vals[inverse]
        return out.reshape(shape + entry)

    def R_matrix_batch(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        """Evaluate a matrix-valued R at arrays of time pairs.

        The ``iso_R=False`` counterpart of :meth:`R_time_batch`, used by the
        batched integrators: ``out[s] = Θ(t1[s] − t2[s]) R_time(t1[s], t2[s])``
        with the strict test of :meth:`R_time_batch` (Itô: equal times give
        0).  ``model.R_time`` is called only on causal pairs, and once per
        distinct pair: on a tensor-product Gauss-Legendre grid an R that
        touches a fixed external point takes as many distinct values as its
        other end has nodes, so most samples reuse an evaluation.

        Args:
            t1: Array of shape ``(n,)``, the left times.
            t2: Array of shape ``(n,)``, the right times.

        Returns:
            Array of shape ``(n, N, N)``.

        Raises:
            ValueError: if ``R_time`` does not return an ``(N, N)`` matrix.
        """
        N = int(self.model.n_components)
        return self._causal_R(t1, t2, (N, N))

    # ------------------------------------------------------------------ #
    # Quadrature-method resolution and kernel structure
    # ------------------------------------------------------------------ #

    def _direct_is_overridden(self) -> bool:
        """Whether a subclass replaced :meth:`_C_value_direct` (legacy
        extension point).  Such a cache computes C its own way: no
        quadrature is chosen for it and no quadrature kwargs are passed,
        so overrides with the original 4-argument signature keep working.
        """
        return type(self)._C_value_direct is not PropagatorCache._C_value_direct

    @property
    def c_method_resolved(self) -> str:
        """The method of the last table build, from the rules of its pairs.

        ``'closed_form'`` when a C callable replaces the quadrature,
        ``'custom'`` when a subclass overrides :meth:`_C_value_direct`, else
        what :func:`_combined_c_rule` makes of the last table.  With no table,
        or a lazy one that has decided nothing yet, the setting: ``'auto'``
        means the rule is worked out per position, which is also the answer
        for a table that takes ``dblquad`` at some pairs only.
        """
        if self.c_value_fn is not None:
            return "closed_form"
        if self._direct_is_overridden():
            return "custom"
        return self._reported_rule()[0]

    @property
    def n_gauss_resolved(self) -> int | None:
        """Largest node count of the last table's pairs when the rule is
        Gauss-Legendre, else ``None``.  The count given, under an explicit
        method."""
        if self.c_method_resolved != "gauss_legendre":
            return None
        return self._reported_rule()[1]

    @property
    def graded_levels_resolved(self) -> int | None:
        """Largest depth of the last table's pairs under the graded rule, else
        ``None``.  Pairs of that table that converged at a count stay on it."""
        if self.c_method_resolved != "graded":
            return None
        return self._reported_rule()[1]

    def _reported_rule(self) -> tuple[str, int | None]:
        """:func:`_combined_c_rule` of the last table, or the setting."""
        if self._table_rules:
            return _combined_c_rule(self._table_rules.values(), self.n_gauss)
        gauss = self.c_method == "gauss_legendre"
        return self.c_method, self.n_gauss if gauss else None

    def resolve_c_method(
        self, t_max: float, positions: list | None = None
    ) -> tuple[str, int | None]:
        """The rule of a table reaching ``t_max``, without building it.

        Explicit settings are returned as they are.  ``'auto'`` combines
        (:func:`_combined_c_rule`) the rules of :meth:`_c_rule` at
        ``positions``, a list of ``(n1, n2)`` pairs, by default the origin.

        A subclass may override this to pick the rule of every table, lazy
        ones included.  An override that takes ``(self, t_max)`` is called
        with ``t_max`` alone.
        """
        if self.c_value_fn is not None:
            return "closed_form", None
        if self._direct_is_overridden():
            return "custom", None
        if self.c_method != "auto":
            gauss = self.c_method == "gauss_legendre"
            return self.c_method, self.n_gauss if gauss else None
        pairs = positions or [self._probe_positions()]
        return _combined_c_rule(
            [self._c_rule(n1, n2, t_max) for n1, n2 in pairs], self.n_gauss
        )

    def _c_rule(
        self, n1: Any, n2: Any, t_max: float, warn: bool = True
    ) -> tuple[str, int | None]:
        """The rule of C at positions ``(n1, n2)`` up to ``t_max``.

        The one place the rule is decided under ``'auto'``: a function of
        these arguments, and the memo only saves repeating it.  Full tables ask
        per pair, lazy tables per position, direct cells per cell.  A pair that
        ends in ``dblquad`` warns once, unless ``warn`` is false.  The pairs of
        a kernel that is not separable are independent of one another, and
        those of a separable one share :meth:`_origin_rule` when it has one.
        """
        t_max = float(t_max)  # the probes hand it to R_time
        pair = _pair_key(n1, n2)
        if pair != _pair_key(*self._probe_positions()):
            shared = self._origin_rule(t_max, warn)
            if shared is not None:
                return shared
        memo = self._c_rules
        if len(memo) >= _C_RULE_MEMO_SIZE:
            memo.clear()
            self._c_rule_raised.clear()
        if (pair, t_max) not in memo:
            rule, hidden, raised = self._find_c_rule(n1, n2, t_max)
            self._warn_unchecked(hidden, n1, n2)
            self._record_rule(pair, t_max, rule, raised)
        rule = memo[pair, t_max]
        if warn and rule[0] == "dblquad":
            self._warn_no_rule(n1, n2, t_max)
        return rule

    def _record_rule(
        self, pair: tuple, t_max: float, rule: tuple, raised: str | None
    ) -> None:
        """Keep the rule a search found, and the error that ended it."""
        self._c_rules[pair, t_max] = rule
        if raised is None:
            self._c_rule_raised.pop((pair, t_max), None)
        else:
            self._c_rule_raised[pair, t_max] = raised

    def _origin_rule(self, t_max: float, warn: bool) -> tuple[str, int | None] | None:
        """The rule the origin gives every pair of a separable κ² with no σ²,
        whose pairs differ by a factor.

        ``None`` for any other kernel, and where the origin says nothing about
        the other pairs: the kernel vanishes there, is not finite there, or
        raises there.  Otherwise no rule at the origin means none at any pair,
        and the search is not repeated.
        """
        if self._c_time_shape_depends_on_position():
            return None
        origin = self._probe_positions()
        rule = self._c_rule(*origin, t_max, warn=False)
        if rule == _ANY_C_RULE:
            return None
        if rule[0] == "dblquad":
            if not self._c_is_finite_at(*origin, t_max):
                return None
            if warn:
                self._warn_no_rule(*origin, t_max)
        return rule

    def _c_is_finite_at(self, n1: Any, n2: Any, t_max: float) -> bool:
        """Whether C can be read at these positions at all."""
        try:
            value = _C_value_direct_gl(self.model, n1, t_max, n2, t_max, n_gauss=2)
        except Exception:  # noqa: BLE001 (a user callable)
            return False
        return bool(np.all(np.isfinite(value)))

    def _find_c_rule(
        self, n1: Any, n2: Any, t_max: float
    ) -> tuple[tuple[str, int | None], set, str | None]:
        """Gauss-Legendre at the count that converges, else the graded rule at
        the depth that does, else ``dblquad``; the entries of C the accepted
        comparison could not check; and, for ``dblquad``, the error R, κ² or
        σ² raised that ended the search, as bounded text, or ``None``.

        ``dblquad`` stops when its first samples agree, and those miss a
        ridge or a boundary layer far narrower than the interval, so it can
        return a value off by any factor with a small error estimate.  The
        graded rule puts its panels on those features.

        Whether a rule exists is the only test: the time taken is not
        compared, so the choice does not depend on machine load.
        """
        pair = (n1, n2)
        hidden: set = set()
        raised: list = []
        t_max = float(t_max)
        live = (
            _c_live_entries(self.model, n1, n2, t_max)
            if t_max > float(self.model.t_min)
            else None
        )
        count = select_gl_node_count(
            self.model,
            t_max,
            n_start=self.n_gauss,
            positions=pair,
            unchecked=hidden,
            live=live,
            raised=raised,
        )
        if count is None:
            levels = select_graded_levels(
                self.model,
                t_max,
                positions=pair,
                unchecked=hidden,
                live=live,
                raised=raised,
            )
            rule = ("dblquad", None) if levels is None else ("graded", int(levels))
        elif count == self.n_gauss and not np.any(
            (live or _c_live_entries(self.model, n1, n2, t_max))[0]
        ):
            rule = _ANY_C_RULE
        else:
            rule = ("gauss_legendre", int(count))
        error = _error_text(raised[-1]) if raised and rule[0] == "dblquad" else None
        return rule, hidden, error

    def _warn_unchecked(self, hidden: set, n1: Any, n2: Any) -> None:
        """Warn once that entries of C could not be checked."""
        if hidden and "unchecked" not in self._c_rule_warnings:
            self._c_rule_warnings.add("unchecked")
            warnings.warn(
                f"c_method='auto': the terms of entries {sorted(hidden)} of C "
                f"at {self._shown_pair(n1, n2)} cancel, and their rounding is "
                f"above the tolerance of the convergence check, so these "
                f"entries cannot be checked to that tolerance.",
                stacklevel=3,
            )

    def _cell_rule(self, n1: Any, t1: Any, n2: Any, t2: Any) -> tuple[str, int | None]:
        """:meth:`_c_rule` for one direct evaluation under ``'auto'``.

        Inside a table the horizon is the table's, so the cell is what the
        table holds.  Otherwise it is :func:`_c_rule_horizon`, which depends on
        ``t1`` and ``t2`` alone, shared by all the cells below it.  A kink or a
        kernel past a limit at that horizon says nothing about this cell, so
        when no rule is found there the cell's own horizon decides, each such
        cell paying for all the steps, tens of seconds on a kernel that reaches
        their end.  A table or an explicit ``c_method`` avoids that.
        """
        t1, t2 = float(t1), float(t2)
        t_min = float(self.model.t_min)
        if min(t1, t2) <= t_min:
            return _ANY_C_RULE  # empty domain, C = 0
        t = max(t1, t2)
        if self._table_horizon is not None and t <= self._table_horizon:
            return self._c_rule(n1, n2, self._table_horizon)
        horizon = _c_rule_horizon(t_min, t)
        rule = self._c_rule(n1, n2, horizon, warn=False)
        if rule[0] == "dblquad":
            rule = self._c_rule(n1, n2, t, warn=False)
        if rule[0] == "dblquad":
            self._warn_no_rule(n1, n2, t, horizon)
        return rule

    def _warn_no_rule(
        self, n1: Any, n2: Any, t_max: float, horizon: float | None = None
    ) -> None:
        """Warn once per pair and ``horizon`` that C takes ``dblquad`` there."""
        key = (_pair_key(n1, n2), t_max if horizon is None else horizon)
        if key not in self._c_rule_warnings:
            self._c_rule_warnings.add(key)
            warnings.warn(self._no_rule_message(n1, n2, t_max), stacklevel=3)

    @staticmethod
    def _shown_pair(n1: Any, n2: Any) -> str:
        """The positions, bounded, for a message."""
        a, b = (np.array2string(np.asarray(x), threshold=6) for x in (n1, n2))
        return f"positions ({a}, {b})"

    def _no_rule_message(self, n1: Any, n2: Any, t_max: float) -> str:
        """Why C falls back to ``dblquad``: a search that an error ended was
        not a search that failed to converge, and saying no rule converged
        would send the reader to the wrong cause.  The error is the one the
        search met, kept by :meth:`_record_rule`."""
        where = self._shown_pair(n1, n2)
        # A pair with no search of its own took the origin's rule
        # (:meth:`_origin_rule`), so the origin's search is the one to quote.
        searched = _pair_key(n1, n2)
        if (searched, float(t_max)) not in self._c_rules:
            searched = _pair_key(*self._probe_positions())
        raised = self._c_rule_raised.get((searched, float(t_max)))
        if raised is not None:
            return (
                f"c_method='auto': R, κ² or σ² raised {raised} at {where} "
                f"while the rules of C up to t={t_max:g} were tested, so C falls "
                f"back to dblquad there.  dblquad calls the same R, κ² and σ², "
                f"so it raises the same error where it reaches the same times; "
                f"where it does not, its value is not checked and can be wrong "
                f"by any factor."
            )
        return (
            f"c_method='auto': no Gauss-Legendre rule converged on C at {where} "
            f"up to t={t_max:g}, tensor or graded, so C falls back to dblquad "
            f"there, whose value is not checked and can be wrong by any factor "
            f"when the kernel or a rate is narrow against the horizon.  A kink "
            f"or a step inside the interval also ends here.  Compare with a "
            f"closed form, or shorten the horizon."
        )

    def _c_time_shape_depends_on_position(self) -> bool:
        """Whether a rule converged at one pair of positions can fail at
        another.  A separable κ² with no σ² differs between positions by a
        factor, which its class declares (``_time_shape_is_shared``); any other
        kernel may be narrower somewhere else.  Any σ², a ``ConstantImpulse``
        included, counts as position dependent: its term does not scale with
        the spatial factor of κ², so an entry where the two cancel at one pair
        need not cancel at the origin, and a separable kernel with σ² searches
        once per pair of a full table and once per position a lazy one reads."""
        return self.model.sigma2 is not None or not getattr(
            self.model.kappa2, "_time_shape_is_shared", False
        )

    def _resolve_build_method(
        self, c_method, n_gauss, t_max, positions=None, n_jobs=1
    ) -> tuple[str, int | None]:
        """Set the rules of a table reaching ``t_max`` and return what they
        come to.  Per-build override, then the instance's setting, then
        ``'auto'``.

        Under ``'auto'`` ``n_gauss`` is not a node count: the refinement starts
        from the instance's ``n_gauss``.  An explicit ``'gauss_legendre'``, per
        build or on the instance, uses exactly the count given.  ``positions``
        are the position pairs of a full table's grid, each with its rule.
        """
        self._table_horizon = None
        if c_method == "gauss_legendre":
            rule = c_method, int(self.n_gauss if n_gauss is None else n_gauss)
        elif c_method == "dblquad":
            rule = c_method, None
        elif c_method is not None and c_method != "auto":
            raise ValueError(
                f"c_method must be 'auto', 'dblquad' or 'gauss_legendre'; "
                f"got {c_method!r}"
            )
        elif (
            self.c_method == "auto"
            and self.c_value_fn is None
            and not self._direct_is_overridden()
            and type(self).resolve_c_method is PropagatorCache.resolve_c_method
        ):
            self._table_horizon = float(t_max)
            self._table_rules = self._auto_table_rules(
                float(t_max), positions, n_jobs
            )
            return self._reported_rule()
        else:
            rule = self._resolve_with_positions(t_max, positions)
            if n_gauss is not None and self.c_method == rule[0] == "gauss_legendre":
                rule = rule[0], int(n_gauss)
        self._table_rules = {None: rule}
        return rule

    def _auto_table_rules(
        self, t_max: float, positions: list | None, n_jobs: int = 1
    ) -> dict:
        """The rules of a table under ``'auto'``, by pair.

        A lazy table has no positions yet, and starts from the origin's rule
        when that says anything for the positions to come.  A kernel can raise
        on the scalar probe, be infinite there, or vanish there, and then only
        the positions read have rules, so nothing is warned about here.
        """
        if positions is not None:
            if n_jobs != 1 and self._c_time_shape_depends_on_position():
                self._find_c_rules(t_max, positions, n_jobs)
            return {_pair_key(a, b): self._c_rule(a, b, t_max) for a, b in positions}
        origin = self._probe_positions()
        rule = self._origin_rule(t_max, warn=True)
        if rule is None:
            rule = self._c_rule(*origin, t_max, warn=False)
            if rule == _ANY_C_RULE or rule[0] == "dblquad":
                return {}
        return {_pair_key(*origin): rule}

    def _find_c_rules(self, t_max: float, positions: list, n_jobs: int) -> None:
        """Fill the memo of :meth:`_c_rule` for ``positions`` across ``n_jobs``
        workers.  The warnings stay in this process."""
        from .progress import progress_map

        memo = self._c_rules
        todo = [p for p in positions if (_pair_key(*p), t_max) not in memo]
        found = progress_map(
            lambda pair: self._find_c_rule(*pair, t_max),
            todo,
            "C rules",
            n_jobs=n_jobs,
            unit="pair",
        )
        for pair, (rule, hidden, raised) in zip(todo, found):
            self._warn_unchecked(hidden, *pair)
            self._record_rule(_pair_key(*pair), t_max, rule, raised)

    def _resolve_with_positions(
        self, t_max: float, positions: list | None
    ) -> tuple[str, int | None]:
        """:meth:`resolve_c_method`, leaving ``positions`` out for an override
        whose signature does not take them."""
        try:
            inspect.signature(self.resolve_c_method).bind(t_max, positions)
        except TypeError:
            return self.resolve_c_method(t_max)
        return self.resolve_c_method(t_max, positions)

    def _table_kwargs(self, x1: Any, x2: Any, t_max: float) -> dict:
        """Keywords of ``_C_value_direct`` for a cell at ``(x1, x2)`` of a
        table reaching ``t_max``, whose pair then counts as read by it."""
        if self._table_horizon is None:
            return self._direct_kwargs(*self._table_rules[None])
        rule = self._c_rule(x1, x2, t_max)
        self._table_rules[_pair_key(x1, x2)] = rule
        return self._direct_kwargs(*rule)

    def _direct_kwargs(self, method: str, n_gauss: int | None) -> dict:
        """Keyword arguments for ``_C_value_direct`` under ``method``.

        None for ``dblquad`` on an override of ``_C_value_direct``, or for a
        closed form, so subclasses with the legacy 4-argument signature keep
        working.
        """
        if method in ("gauss_legendre", "graded"):
            return {"method": method, "n_gauss": int(n_gauss or self.n_gauss)}
        if method == "dblquad" and not self._direct_is_overridden():
            return {"method": method}
        return {}

    def _require_even_separation(self, pairs: list[tuple]) -> None:
        """Raise ``ValueError`` when a translation table would give one value
        to the two orientations of a separation.

        Such a table is keyed on ``|x1 − x2|``, which holds for noise even in
        ``x1 − x2``.  A moving pattern ``κ²(x − y − v(t − t'))`` is translation
        invariant and not even: the table returned the orientation it read
        first.  Each pair is read in both orders (:func:`_noise_reads_alike`).
        A closed-form C replaces the noise and a kernel that cannot be read at
        a pair is not judged.
        """
        if _closed_form_c(self) is not None:
            return
        for x1, x2 in pairs:
            try:
                alike = _noise_reads_alike(
                    self.model, (x1, x2), (x2, x1), self._c_table_range
                )
            except Exception:  # noqa: BLE001 (a user callable or a stand-in)
                continue
            if not alike:
                raise ValueError(
                    f"homogeneity='translation' keys C on |x1 - x2|, and the "
                    f"noise reads differently at positions ({x1!r}, {x2!r}) and "
                    f"({x2!r}, {x1!r}) at equal times.  Use "
                    f"homogeneity='general'."
                )

    def _probe_positions(self) -> tuple:
        """``(n1, n2)`` the node-count probe evaluates κ² at."""
        if self.homogeneity == "rotation":
            v = np.array([0.0, 0.0, 1.0])
            return v, v
        return np.asarray(0.0), np.asarray(0.0)

    def _c_source_label(self) -> str:
        """Short description of how C is evaluated, for progress bars."""
        m = self.c_method_resolved
        if m == "closed_form":
            return "closed form"
        if m == "custom":
            return "custom C"
        if m == "gauss_legendre":
            return f"Gauss-Legendre n={self.n_gauss_resolved}"
        if m == "graded":
            return f"graded Gauss-Legendre, {self.graded_levels_resolved} levels"
        if m == "auto":
            return "rule chosen per position"
        return "dblquad"

    def _lazy_spatial_factor(self) -> Callable | None:
        """``r ↦ κ_x(r)`` when C factorises as ``κ_x(r) · C(0; t1, t2)``.

        True for a separable translation-invariant ``kappa2`` (the L1
        :class:`~sft_wick.workflow.specs.SeparableTranslation`) and for a
        closed form that declares ``separable_translation``, such as
        :class:`~sft_wick.workflow.closed_forms.ClosedFormC`.  A white-noise
        term of either adds an ``r``-independent term that would break the
        scaling.  Any other C is opaque.
        """
        if self._direct_is_overridden():
            return None
        c_fn = _closed_form_c(self)
        if c_fn is not None:
            source, sigma2 = c_fn, getattr(c_fn, "sigma2", None)
        else:
            source, sigma2 = self.model.kappa2, self.model.sigma2
        if (
            sigma2 is None
            and getattr(source, "separable_translation", False)
            and hasattr(source, "spatial_factor")
        ):
            return source.spatial_factor
        return None

    def _kappa2_has_diagonal_cusp(
        self, positions: tuple | None = None, span: float | None = None
    ) -> bool:
        """Whether ``κ²(λ1, λ2)`` has a derivative jump on ``λ1 = λ2``.

        Decides whether the dblquad path splits the rectangle at the
        diagonal.  Splitting is what makes a stationary ``|λ1 − λ2|``
        kernel fast and accurate (7 ms and 1e-10 instead of 74-245 ms and
        2e-6), but on a SMOOTH kernel the three variable-limit pieces cost
        scipy 10-250× more evaluations than the one rectangle (measured
        0.2 ms → 8-50 ms), so the choice must be made per kernel.  The
        built-in kernels say so themselves (``has_diagonal_cusp``); any
        other callable is probed once: the jump ``|κ'(0⁺) − κ'(0⁻)|`` is
        estimated from one-sided differences at two step sizes and called
        a cusp when it does not shrink with the step.  A kernel that
        raises on the probe is treated as having a cusp (the safe side).

        The probe reads κ² at one pair of positions, ``positions`` or else
        :meth:`_probe_positions`, and at times across a range of length
        ``span``.  A kernel can vanish or be smooth at the origin and not
        elsewhere, or on part of a range, so the answer is kept per pair
        and per range.
        """
        flag = getattr(self.model.kappa2, "has_diagonal_cusp", None)
        if flag is not None:
            return bool(flag)
        n1, n2 = self._probe_positions() if positions is None else positions
        span = _CUSP_PROBE_SPAN if span is None else float(span)
        key = (_position_key(n1), _position_key(n2), span)
        memo = self.__dict__.setdefault("_cusp_memo", {})
        if len(memo) >= _C_KINK_MEMO_SIZE:
            memo.clear()
        if key not in memo:
            memo[key] = bool(_probe_diagonal_cusp(self.model, (n1, n2), span))
        return memo[key]

    def _c_time_symmetric(self) -> bool:
        """Whether a C table is fixed by its cells with ``t2 ≥ t1``, through
        ``C_ab(t1, t2) = C_ba(t2, t1)``.

        This takes ``κ_ij(s1, s2) = κ_ji(s2, s1)``, which no finite set of
        reads can establish for a callable: a kernel can be one-sided only
        past the lags read, or at a separation of its own.  So it is taken
        from the kernel's own ``symmetric_in_time``, which the built-in even
        temporal kernels declare, or from a closed form that declares
        ``separable_translation`` (for ``C_aa`` only).  With ``diag_C=False``
        a σ² has to declare ``is_symmetric`` as well.
        """
        if self._direct_is_overridden():
            return False
        c_fn = _closed_form_c(self)
        if c_fn is not None:
            return self.model.diag_C and bool(
                getattr(c_fn, "separable_translation", False)
            )
        return bool(getattr(self.model.kappa2, "symmetric_in_time", False)) and (
            self.model.diag_C
            or self.model.sigma2 is None
            or getattr(self.model.sigma2, "is_symmetric", False)
        )

    # ------------------------------------------------------------------ #
    # Table layout: C_aa under diag_C, every C_ab otherwise
    # ------------------------------------------------------------------ #

    def _c_table_entries(self) -> tuple:
        """``(a, b)`` pairs every table of this cache stores: ``C_aa`` under
        ``model.diag_C``, every ``C_ab`` otherwise (see
        :func:`_c_table_entries`)."""
        return _c_table_entries(self.model.n_components, self.model.diag_C)

    def _assemble_C(self, columns: list, n: int) -> np.ndarray:
        """One ``(n,)`` column per stored entry → ``(n, N)`` diagonals under
        ``diag_C``, ``(n, N, N)`` matrices otherwise."""
        N = self.model.n_components
        entries = self._c_table_entries()
        if self.model.diag_C:
            out = np.empty((n, N))
            for (a, _b), col in zip(entries, columns):
                out[:, a] = col
            return out
        out = np.empty((n, N, N))
        for (a, b), col in zip(entries, columns):
            out[:, a, b] = col
        return out

    def _need_earlier_first(self, mode: str) -> bool:
        """Whether a table of ``mode`` stores the earlier-first half ``E``.

        Where ``C_ab(t1, t2) = C_ba(t2, t1)`` is declared, the earlier-first
        value at a swap-invariant spatial key (translation's ``r``, rotation's
        ``cos``) is the later-first value of the transposed entry, so one grid
        covers both orders.  General keys are not swap invariant.
        """
        return mode == "general" or not self._c_time_symmetric()

    def _tabulate(
        self,
        shape: tuple,
        cells: list,
        t_max: float,
        desc: str,
        n_jobs: int,
    ) -> list:
        """Evaluate ``_C_value_direct`` at ``cells``; one grid of ``shape``
        per stored entry.

        Args:
            cells: ``(index, x1, t1, x2, t2)`` per evaluated cell.
            t_max: The table's horizon, for the rule of each pair
                (:meth:`_table_kwargs`).
        """
        from .progress import progress_map

        entries = self._c_table_entries()

        def _point(args):
            idx, x1, t1, x2, t2 = args
            C_mat = np.asarray(
                _C_value_direct_on_copies(
                    self, x1, t1, x2, t2, **self._table_kwargs(x1, x2, t_max)
                )
            )
            return idx, np.array([C_mat[a, b] for a, b in entries])

        results = progress_map(_point, cells, desc, n_jobs=n_jobs, unit="cell")
        grids = [np.zeros(shape) for _ in entries]
        for idx, cvec in results:
            for k in range(len(entries)):
                grids[k][idx] = cvec[k]
        return grids

    def _full_table(
        self,
        t_max: float,
        n_grid_t: int,
        axes: tuple,
        positions_at: Callable[[tuple], tuple],
        mode: str,
        grid_name: str,
        n_jobs: int,
    ) -> list:
        """One :class:`_MinLagInterp` per stored entry of a full table on
        ``(s, u, *axes)``.

        The later-first ``L_ab = C_ab(x1, s+u; x2, s)`` has its kink on
        ``u = 0``, a grid boundary, so no cell straddles it.  The earlier
        half ``E_ab`` is built only where :meth:`_need_earlier_first` says so.
        Otherwise ``E_ab = L_ba``.

        Args:
            axes: the spatial axes.
            positions_at: spatial index ``->`` ``(x1, x2)``.
            mode: the table's kind, ``translation``, ``rotation`` or ``general``.
            grid_name: names the grid on the progress bar.
        """
        from scipy.interpolate import RegularGridInterpolator

        ts = np.linspace(self.model.t_min, t_max, n_grid_t)
        us = np.linspace(0.0, t_max - self.model.t_min, n_grid_t)
        layout = _MinLagGrid(ts)
        shape = (len(layout.points), *(len(axis) for axis in axes))

        # At ``later == earlier`` the two halves hold the same cell.
        diagonal = [i for i, (later, earlier) in enumerate(layout.points)
                    if later == earlier]

        def half(later_first: bool) -> list:
            cells = []
            for i, (later, earlier) in enumerate(layout.points):
                if not later_first and later == earlier:
                    continue
                t1, t2 = (later, earlier) if later_first else (earlier, later)
                for k in np.ndindex(*shape[1:]):
                    x1, x2 = positions_at(k)
                    cells.append(((i, *k), x1, t1, x2, t2))
            return self._tabulate(
                shape,
                cells,
                t_max,
                f"C table ({self._c_source_label()}, full {grid_name}-grid)",
                n_jobs,
            )

        grids = half(True)
        egrids = half(False) if self._need_earlier_first(mode) else None
        for e, g in zip(egrids or (), grids):
            e[diagonal] = g[diagonal]

        # Default 'linear' (set in PropagatorCache.__init__): tensor-product
        # cubic on a steeply decaying C produces sign flips and O(1)
        # overshoot in the tail; linear is monotone and safe. Users with
        # smooth C and dense grids can opt into cubic via
        # interp_method='cubic'. See tests/test_evaluate_interpolation_accuracy.py.
        def interpolators(halves: list) -> list:
            return [
                RegularGridInterpolator(
                    (ts, us, *axes),
                    layout.complete(g),
                    bounds_error=False,
                    fill_value=None,
                    method=self.interp_method,
                )
                for g in halves
            ]

        entries = self._c_table_entries()
        where = {e: k for k, e in enumerate(entries)}
        raw = interpolators(grids)
        if egrids is None:
            return [
                _MinLagInterp(raw[where[(a, b)]], raw[where[(b, a)]])
                for a, b in entries
            ]
        eraw = interpolators(egrids)
        return [
            _MinLagInterp(raw[where[(a, b)]], eraw[where[(a, b)]])
            for a, b in entries
        ]

    # ------------------------------------------------------------------ #
    # Spatial-aware extension: (t, x) coordinates via homogeneity modes
    # ------------------------------------------------------------------ #

    def precompute_C_table_translation(
        self,
        t_max: float,
        n_grid_t: int = 60,
        r_max: float | None = None,
        n_grid_r: int | None = None,
        n_jobs: int = 1,
        c_method: str | None = None,
        n_gauss: int | None = None,
    ) -> None:
        """Enable spatial support under the translation-invariance
        assumption:

            ``C(t1, t2; x1, x2)`` depends only on ``|x1 − x2|``.

        Two operating modes depending on whether ``r_max`` and
        ``n_grid_r`` are supplied:

        **Lazy mode (default, recommended for fixed-point moments)**
            ``r_max=None`` and ``n_grid_r=None``.  No r-grid is
            pre-computed.  When :meth:`C_at_batch` encounters a new
            ``r = |x1−x2|`` value it builds and caches a 2-D
            ``(t1, t2)`` spline on demand.  For a moment calculation
            at fixed external points ``(x_1, …, x_n)`` there are at
            most ``O(n²)`` distinct r-values, so lazy mode does far
            less work than a densely-sampled r-grid, particularly when
            the underlying ``_C_value_direct`` is
            ``scipy.integrate.dblquad`` (expensive).

        **Full-grid mode (right for sweeping r curves)**
            Both ``r_max`` and ``n_grid_r`` provided.  A 3-D
            ``(t1, t2, r)`` spline is built up-front over the full
            range, giving O(1) evaluation per query, faster once
            ``n_distinct_r >> n_grid_r``.  Appropriate when the user
            will plot ``C`` or a derived observable as a function of
            r over a continuous range.

        Args:
            t_max: upper bound of the (t1, t2) time grid.
            n_grid_t: grid size along each time axis.
            r_max: upper bound of the r = ``|x1-x2|`` grid
                (full-grid mode); :meth:`C_at_batch` refuses a separation
                past it.  ``None`` ⇒ lazy mode.
            n_grid_r: grid size along r (full-grid mode).
                ``None`` ⇒ lazy mode.
            n_jobs: parallel workers for grid-point ``_C_value_direct``
                evaluations.  ``1`` serial, ``-1`` all cores via
                :mod:`joblib` (``loky`` backend).  Applies in **both**
                full-grid mode (fans out across the 3-D ``(t1, t2, r)``
                grid) **and** lazy mode (fans out across each on-demand
                2-D ``(t1, t2)`` build; each new r pays a ~1 s worker
                startup cost).
            c_method: Quadrature method passed to
                :meth:`_C_value_direct` for every grid-point
                evaluation, in both full-grid and lazy mode.  ``None``
                (default) uses the cache instance's setting, resolving
                ``'auto'`` for this ``t_max`` (see
                :meth:`resolve_c_method`); ``'dblquad'`` /
                ``'gauss_legendre'`` override it for this build.
            n_gauss: Per-dimension GL node count for an explicit
                ``'gauss_legendre'``; ``None`` uses the instance's value.
                Not used under ``'auto'``, where the count is the
                converged one.

        Requires ``self.homogeneity == 'translation'``.

        Side effect: sets either ``_c_translation_splines`` (full)
        or ``_lazy_translation`` (lazy).

        Raises:
            ValueError: if the noise is not even in ``x1 − x2`` at a
                separation the table reads, full or lazy; see
                :meth:`_require_even_separation`.
        """
        if self.homogeneity != "translation":
            raise RuntimeError(
                f"precompute_C_table_translation requires "
                f"homogeneity='translation', got {self.homogeneity!r}"
            )

        # A float, not the caller's object: the table keeps it as its range
        # and the method probe hands it to R_time (select_gl_node_count).
        t_max = float(t_max)
        m = self.model
        t_min = m.t_min
        self._c_table_range = (t_min, t_max)
        if r_max is not None and n_grid_r is not None:
            self._require_even_separation(
                [(0.0, r) for r in np.linspace(0.0, float(r_max), n_grid_r)]
            )
        self._resolve_build_method(
            c_method,
            n_gauss,
            t_max,
            None
            if r_max is None or n_grid_r is None
            else [
                (np.asarray(0.0), np.asarray(r))
                for r in np.linspace(0.0, float(r_max), n_grid_r)
            ],
            n_jobs,
        )

        # Lazy mode: defer per-r spline construction to first query.
        if r_max is None or n_grid_r is None:
            self._lazy_translation = _LazyTimeSplineCache(
                self,
                t_max,
                n_grid_t,
                mode="translation",
                n_jobs=n_jobs,
            )
            return

        # Full-grid mode: build the 3-D (t1, t2, r) spline now.
        rs = np.linspace(0.0, r_max, n_grid_r)
        # TODO(d-dim): replace scalar r with np.ndarray x_diff
        self._c_translation_splines = self._full_table(
            t_max,
            n_grid_t,
            (rs,),
            lambda k: (np.asarray(0.0), np.asarray(rs[k[0]])),
            "translation",
            "r",
            n_jobs,
        )
        self._c_translation_r_range = (0.0, r_max)

    def precompute_C_table_rotation(
        self,
        t_max: float,
        n_grid_t: int = 60,
        n_grid_cos: int | None = None,
        n_jobs: int = 1,
        c_method: str | None = None,
        n_gauss: int | None = None,
    ) -> None:
        """Enable spatial support under the rotation-invariance
        assumption:

            ``C(t1, t2; x1, x2)`` depends only on ``x1 · x2``, so it is even
            under swapping the points.  This is not checked: a kernel with a
            term in the signed angle takes one value for both orientations,
            and on a lazy table the orientation read first.

        Appropriate when ``x`` is a unit direction vector (e.g. on
        the sphere).  The effective spatial parameter is
        ``cos θ ∈ [−1, 1]``.

        Two modes, analogous to
        :meth:`precompute_C_table_translation`:

        - Lazy (``n_grid_cos=None``): 2-D ``(t1, t2)`` splines built
          on demand per distinct ``cos θ`` value.  Best when only
          a few ``cos θ`` values arise (e.g. a fixed-set moment
          calculation).
        - Full-grid (``n_grid_cos`` integer): 3-D
          ``(t1, t2, cos)`` spline over the whole ``cos ∈ [−1, 1]``
          range.  Best for angular-correlation sweeps.

        Under the hood, the reference evaluation at a specific
        ``cos θ`` uses two representative unit vectors
        ``e_1 = (1, 0, …)`` and ``e_2`` chosen so that
        ``e_1 · e_2 = cos θ`` (a 2-D rotation of ``e_1``).

        Args:
            n_jobs: parallel workers for grid-point
                ``_C_value_direct`` evaluations; applies in **both**
                full-grid (fans out across the 3-D ``(t1, t2, cos)``
                grid) and lazy mode (fans out across each on-demand
                2-D build).  ``1`` serial, ``-1`` all cores via
                :mod:`joblib`.
            c_method: ``None`` (default, the instance's setting),
                ``'dblquad'`` or ``'gauss_legendre'``, forwarded to
                :meth:`_C_value_direct` for every full-grid point.  See
                :meth:`precompute_C_table_translation` for details.
            n_gauss: Per-dim GL node count for an explicit
                ``c_method='gauss_legendre'``; ``None`` uses the
                instance's value.  Not used under ``'auto'``.

        Requires ``self.homogeneity == 'rotation'``.
        """
        if self.homogeneity != "rotation":
            raise RuntimeError(
                f"precompute_C_table_rotation requires "
                f"homogeneity='rotation', got {self.homogeneity!r}"
            )

        t_max = float(t_max)  # see precompute_C_table_translation
        m = self.model
        t_min = m.t_min
        self._c_table_range = (t_min, t_max)

        def _rep_vectors(cos_val):
            """Return a pair of unit vectors with dot product cos_val.

            We use the 2-D representation ``e_1 = (1, 0)`` and
            ``e_2 = (cos, sin)`` where ``sin = √(1 − cos²)``.  This
            is sufficient when ``kappa2`` depends only on ``x1·x2``
            (rotation invariance makes the ambient dimension
            irrelevant).
            """
            c = float(cos_val)
            c = max(-1.0, min(1.0, c))
            s = float(np.sqrt(max(0.0, 1.0 - c * c)))
            return np.array([1.0, 0.0]), np.array([c, s])

        self._resolve_build_method(
            c_method,
            n_gauss,
            t_max,
            None
            if n_grid_cos is None
            else [_rep_vectors(c) for c in np.linspace(-1.0, 1.0, n_grid_cos)],
            n_jobs,
        )

        if n_grid_cos is None:
            self._lazy_rotation = _LazyTimeSplineCache(
                self,
                t_max,
                n_grid_t,
                mode="rotation",
                n_jobs=n_jobs,
            )
            return

        coses = np.linspace(-1.0, 1.0, n_grid_cos)
        reps = [_rep_vectors(c) for c in coses]
        self._c_rotation_splines = self._full_table(
            t_max,
            n_grid_t,
            (coses,),
            lambda k: reps[k[0]],
            "rotation",
            "cos",
            n_jobs,
        )

    def precompute_C_table_general(
        self,
        t_max: float,
        n_grid_t: int = 40,
        x_max: float | None = None,
        n_grid_x: int | None = None,
        n_jobs: int = 1,
        c_method: str | None = None,
        n_gauss: int | None = None,
    ) -> None:
        """Pre-compute C with no spatial symmetry assumed.

        Two modes:

        - Lazy (``x_max=None`` or ``n_grid_x=None``): 2-D
          ``(t1, t2)`` splines built on demand per distinct
          ``(x1, x2)`` pair.  Best for fixed-point moments where
          only a handful of pairs occur.
        - Full-grid: 4-D ``(t1, t2, x1, x2)`` interpolator over the
          pre-allocated grid.  Build cost is
          ``n_grid_t² · n_grid_x²`` independent ``_C_value_direct``
          calls; strongly consider ``n_jobs=-1``.  :meth:`C_at_batch`
          refuses a position past ``x_max`` on either side.

        Args:
            n_jobs: parallel workers for ``_C_value_direct``
                evaluations; applies in **both** full-grid (fans out
                across the 4-D grid) and lazy mode (fans out across
                each on-demand 2-D build).  ``1`` serial, ``-1`` all
                cores via :mod:`joblib`.
            c_method: ``None`` (default, the instance's setting),
                ``'dblquad'`` or ``'gauss_legendre'``, forwarded to
                :meth:`_C_value_direct` for every full-grid point.  See
                :meth:`precompute_C_table_translation` for details.
            n_gauss: Per-dim GL node count for an explicit
                ``c_method='gauss_legendre'``; ``None`` uses the
                instance's value.  Not used under ``'auto'``.

        Requires ``self.homogeneity == 'general'``.
        """
        if self.homogeneity != "general":
            raise RuntimeError(
                f"precompute_C_table_general requires "
                f"homogeneity='general', got {self.homogeneity!r}"
            )

        t_max = float(t_max)  # see precompute_C_table_translation
        m = self.model
        t_min = m.t_min
        self._c_table_range = (t_min, t_max)
        self._resolve_build_method(
            c_method,
            n_gauss,
            t_max,
            None
            if x_max is None or n_grid_x is None
            else [
                (np.asarray(a), np.asarray(b))
                for a in np.linspace(-float(x_max), float(x_max), n_grid_x)
                for b in np.linspace(-float(x_max), float(x_max), n_grid_x)
            ],
            n_jobs,
        )

        if x_max is None or n_grid_x is None:
            self._lazy_general = _LazyTimeSplineCache(
                self,
                t_max,
                n_grid_t,
                mode="general",
                n_jobs=n_jobs,
            )
            return

        xs = np.linspace(-x_max, x_max, n_grid_x)
        # TODO(d-dim): x1, x2 would be vectors
        self._c_general_interpolators = self._full_table(
            t_max,
            n_grid_t,
            (xs, xs),
            lambda k: (np.asarray(xs[k[0]]), np.asarray(xs[k[1]])),
            "general",
            "x",
            n_jobs,
        )
        self._c_general_axes = (
            np.linspace(t_min, t_max, n_grid_t),
            np.linspace(0.0, t_max - t_min, n_grid_t),
            xs,
            xs,
        )

    def _C_value_direct(
        self,
        n1: Any,
        t1: float,
        n2: Any,
        t2: float,
        method: str | None = None,
        n_gauss: int | None = None,
    ) -> np.ndarray:
        """Direct quadrature evaluation of C without consulting any cache.

        Needed so the ``precompute_C_table_{translation,rotation,general}``
        builders aren't contaminated by the spline cache they are
        themselves populating.

        When ``model.sigma2`` is provided, the full κ² is
        ``κ²_smooth + δ(t1−t2) · σ²(t; x1, x2)``.  The δ collapses one
        time integral so C picks up an extra 1-D piece::

            C_white[a,b] = ∫_{t_min}^{min(t1,t2)} R(t1,τ) σ²[a,b](τ;
                            n1, n2) R(t2,τ) dτ

        which is added to the usual 2-D dblquad result.

        Args:
            method: ``'dblquad'``, ``'gauss_legendre'`` or ``'graded'``.
                ``None`` (default) uses :attr:`c_method`.
            n_gauss: Override per-dim node count for the Gauss-Legendre
                method, or depth of the graded one.  ``None`` (default)
                uses :attr:`n_gauss` or the first of :data:`_GRADED_LEVELS`.

        With no method, the instance's explicit setting stands, and ``'auto'``
        takes :meth:`_cell_rule`, or Gauss-Legendre at ``n_gauss`` when a count
        was given.  The cells of a build name their method and count, so a
        per-build method reaches them alone, whether the build ended or not.
        """
        c_fn = _closed_form_c(self)
        if c_fn is not None:
            # User-supplied C replaces the whole quadrature construction.
            return _closed_form_value(c_fn(n1, t1, n2, t2), t1, t2)

        m = self.model
        N = m.n_components
        t_min = m.t_min
        C_mat = np.zeros((N, N))
        if t1 <= t_min or t2 <= t_min:
            # No noise has accumulated.  An R_time need not vanish for
            # t_left < t_right (see PropagatorModel), so the rules below would
            # return the integral of a backward domain.
            return C_mat

        if method is None or method == "auto":
            if self.c_method != "auto":
                method = self.c_method
            elif n_gauss is not None:
                method = "gauss_legendre"
            else:
                method, n_gauss = self._cell_rule(n1, t1, n2, t2)
        if method == "gauss_legendre":
            return _C_value_direct_gl(
                self.model,
                n1,
                t1,
                n2,
                t2,
                n_gauss=int(n_gauss or self.n_gauss),
            )
        if method == "graded":
            return _C_value_direct_gl(
                self.model,
                n1,
                t1,
                n2,
                t2,
                graded_levels=int(n_gauss or _GRADED_LEVELS[0]),
            )
        if method != "dblquad":
            raise ValueError(
                f"unknown c_method {method!r}; expected 'dblquad', "
                f"'gauss_legendre' or 'graded'"
            )

        from scipy.integrate import dblquad, quad as _quad

        # ``R`` is a full matrix whenever the linear operator is not diagonal
        # in the chosen component basis.  Contract
        #     C_{ab} = sum_{c,d} R_{ac}(t1,l1) kappa_{cd}(l1,l2) R_{bd}(t2,l2)
        # i.e. row ``a`` of R(t1,·) against kappa against row ``b`` of R(t2,·).
        # Using only R[a,a] is correct solely for diagonal R.
        def _row(t_obs: float, lam: float, idx: int) -> np.ndarray:
            rt = m.R_time(t_obs, lam)
            if m.iso_R:
                row = np.zeros(N)
                row[idx] = float(rt)
                return row
            return np.asarray(rt, dtype=float)[idx, :]

        # Integrate the rectangle ``[t_min, t1] × [t_min, t2]`` as the three
        # cusp-free pieces used by the Gauss-Legendre path (the two
        # triangles of the square ``[t_min, min(t1,t2)]²`` on either side
        # of ``λ1 = λ2``, plus the strip outside it) rather than in one
        # go.  Stationary κ² has a ``|λ1 − λ2|`` cusp on that diagonal,
        # and an adaptive rule straddling it both stalls (per-call cost
        # rising from 74 to 245 ms with ``t`` on the demo1 kernel) and
        # reports roundoff: measured 2e-6 relative error at
        # ``epsrel=1e-10`` against the closed form, versus 1e-10 for the
        # split.  The union of the pieces is the same rectangle, so a κ²
        # without a cusp is unaffected.
        def _split_dblquad(integrand) -> float:
            span = _c_rule_horizon(t_min, max(t1, t2)) - t_min
            if not self._kappa2_has_diagonal_cusp((n1, n2), span):
                val, _ = dblquad(integrand, t_min, t1, t_min, t2, **self.quad_opts)
                return val
            t_d = min(t1, t2)
            total, _ = dblquad(
                integrand,
                t_min,
                t_d,
                lambda l1: t_min,
                lambda l1: l1,
                **self.quad_opts,
            )
            val, _ = dblquad(
                integrand,
                t_min,
                t_d,
                lambda l1: l1,
                lambda l1: t_d,
                **self.quad_opts,
            )
            total += val
            if t1 > t2:
                val, _ = dblquad(
                    integrand,
                    t_d,
                    t1,
                    lambda l1: t_min,
                    lambda l1: t_d,
                    **self.quad_opts,
                )
                total += val
            elif t2 > t1:
                val, _ = dblquad(
                    integrand,
                    t_min,
                    t_d,
                    lambda l1: t_d,
                    lambda l1: t2,
                    **self.quad_opts,
                )
                total += val
            return total

        if m.diag_C:
            for a in range(N):

                def integrand(lam2: float, lam1: float, _a: int = a) -> float:
                    kappa_mat = np.asarray(
                        _kappa2_at(m, n1, lam1, n2, lam2), dtype=float
                    )
                    return float(_row(t1, lam1, _a) @ kappa_mat @ _row(t2, lam2, _a))

                C_mat[a, a] = _split_dblquad(integrand)
        else:
            for a in range(N):
                for b in range(N):

                    def integrand(
                        lam2: float, lam1: float, _a: int = a, _b: int = b
                    ) -> float:
                        kappa_mat = np.asarray(
                            _kappa2_at(m, n1, lam1, n2, lam2), dtype=float
                        )
                        return float(
                            _row(t1, lam1, _a) @ kappa_mat @ _row(t2, lam2, _b)
                        )

                    C_mat[a, b] = _split_dblquad(integrand)

        # --- White-noise (δ-correlated) component: 1-D integral ---
        if m.sigma2 is not None:
            t_upper = min(t1, t2)
            if m.diag_C:
                for a in range(N):

                    def w_integrand(tau: float, _a: int = a) -> float:
                        sig = np.asarray(_sigma2_at(m, n1, tau, n2), dtype=float)
                        return float(_row(t1, tau, _a) @ sig @ _row(t2, tau, _a))

                    wval, _ = _quad(
                        w_integrand,
                        t_min,
                        t_upper,
                        **self.quad_opts,
                    )
                    C_mat[a, a] += wval
            else:
                for a in range(N):
                    for b in range(N):

                        def w_integrand(tau: float, _a: int = a, _b: int = b) -> float:
                            sig = np.asarray(_sigma2_at(m, n1, tau, n2), dtype=float)
                            return float(_row(t1, tau, _a) @ sig @ _row(t2, tau, _b))

                        wval, _ = _quad(
                            w_integrand,
                            t_min,
                            t_upper,
                            **self.quad_opts,
                        )
                        C_mat[a, b] += wval

        return C_mat

    def C_at_batch(
        self,
        t1: np.ndarray,
        t2: np.ndarray,
        x1: np.ndarray | float,
        x2: np.ndarray | float,
    ) -> np.ndarray:
        """Batch-evaluate C at arbitrary ``(t1, t2, x1, x2)`` arrays.

        Dispatches on :attr:`homogeneity`:

        - ``'translation'``: C depends only on ``|x1 − x2|``.  Uses
          the 3-D ``(t1, t2, r)`` full-grid spline if built via
          ``precompute_C_table_translation(r_max=..., n_grid_r=...)``;
          otherwise uses a lazy per-r 2-D spline cache; otherwise
          (legacy 2-D spline only) falls back to
          :meth:`C_diagonal_batch`, where x is ignored.
        - ``'rotation'``: C depends only on ``x1 · x2 / (|x1| |x2|)``.
          Uses the 3-D ``(t1, t2, cos)`` spline or per-cos lazy cache.
        - ``'general'``: 4-D ``(t1, t2, x1, x2)`` spline, or a lazy
          per-pair 2-D cache.

        Args:
            t1, t2: Time arrays of shape ``(n,)``.
            x1, x2: Spatial coordinates, each one of: a scalar (one
                point for every sample); shape ``(n,)`` (one scalar
                position per sample); shape ``(d,)`` with ``d != n``
                (one fixed point in d dimensions); shape ``(n, d)``
                (one point per sample).  A 1-D array of length ``n`` is
                always read as ``n`` scalar positions, so a fixed
                d-vector on a batch of ``n == d > 1`` samples must be
                passed as ``(n, d)``, which
                :func:`_C_at_fixed_positions` does.

        Returns:
            ``(n, N)``, the diagonal C values per component, when
            ``model.diag_C``; ``(n, N, N)``, every entry, otherwise.

        Raises:
            ValueError: if a sample is outside the table it reads: a time
                outside ``[t_min, t_max]``, a separation past ``r_max`` or a
                position past ``x_max`` on a full spatial grid.  A value
                within rounding of a limit is read at the limit.
        """
        # Read-only views: the time-table fallback hands these arrays to a
        # C_diagonal_batch or C_matrix_batch that a subclass may override,
        # and the table branches to _lazy_lookup and
        # _closed_form_at_batch_diag (see _read_only_view).
        t1 = _read_only_view(np.atleast_1d(np.asarray(t1)))
        t2 = _read_only_view(np.atleast_1d(np.asarray(t2)))
        # The positions reach _lazy_lookup or _closed_form_at_batch_diag as
        # the caller's arrays (one object for both ends when the caller passed
        # one).  An override of either may write into them, so it gets
        # copies.  The integrators pass copies already (_C_at_fixed_positions)
        # and the package's own delegates only read them.
        if _overrides(
            self,
            PropagatorCache._lazy_lookup,
            PropagatorCache._closed_form_at_batch_diag,
        ):
            x1, x2 = _own_copy(x1), _own_copy(x2)

        # Closed-form-only: skip every spline and call the user's
        # analytical C function directly. Used by
        # ``Propagators(c_closed_form=..., c_closed_form_only=True)``;
        # gives machine-precision agreement with the analytical C
        # (no interpolation error). The vectorised contract is the
        # fast path; the per-sample fallback runs a Python loop and
        # is only practical for small sweeps.
        if self._closed_form_only:
            return self._closed_form_at_batch_diag(t1, t2, x1, x2)
        self._refuse_times_past_table(t1, t2)
        n = len(t1)

        x1_b = _positions_per_sample(x1, n)
        x2_b = _positions_per_sample(x2, n)

        if self.homogeneity == "translation":
            # Full-grid 3-D (t1, t2, r) spline takes precedence.
            if self._c_translation_splines is not None:
                r = _separation_batch(x1_b, x2_b)
                if self._c_translation_r_range is not None:
                    _refuse_past_table(
                        "r",
                        r,
                        *self._c_translation_r_range,
                        "Rebuild the propagators with a larger r_max, or "
                        "leave r_max out for a lazy table.",
                    )
                pts = np.stack([t1, t2, r], axis=-1)
                return self._assemble_C(
                    [itp(pts) for itp in self._c_translation_splines], n
                )
            if self._lazy_translation is not None:
                return self._lazy_lookup(
                    self._lazy_translation,
                    t1,
                    t2,
                    x1_b,
                    x2_b)
            # Fall back to legacy 2-D (t1, t2) spline, where x is
            # effectively ignored (r=0 assumed).  Matches behaviour
            # of users who only called the legacy
            # :meth:`precompute_C_table`.
            return _time_table_C_batch(self, t1, t2)

        if self.homogeneity == "rotation":
            if self._c_rotation_splines is not None:
                # cos per sample
                cosv = np.empty(n)
                for k in range(n):
                    cosv[k] = _rotation_cos(x1_b[k], x2_b[k])
                pts = np.stack([t1, t2, cosv], axis=-1)
                return self._assemble_C(
                    [itp(pts) for itp in self._c_rotation_splines], n
                )
            if self._lazy_rotation is not None:
                return self._lazy_lookup(
                    self._lazy_rotation,
                    t1,
                    t2,
                    x1_b,
                    x2_b)
            raise RuntimeError(
                "homogeneity='rotation' but no rotation table has "
                "been built — call precompute_C_table_rotation() first"
            )

        # homogeneity == "general"
        if self._c_general_interpolators is not None:
            # Full-grid general mode is built over a 1-D x axis
            # (precompute_C_table_general uses ``np.linspace(-x_max,
            # x_max, n_grid_x)``). Extending the grid to d dimensions
            # would yield a (2 + 2d)-D spline whose build cost scales
            # as ``n_grid_x ** (2d)``, exponentially expensive even
            # for d=2. Reject vector inputs in full-grid mode and
            # direct the user to lazy mode, which already supports
            # d-dim via dict-keyed memoisation.
            if x1_b.ndim > 1 or x2_b.ndim > 1:
                raise NotImplementedError(
                    "general-mode full-grid C tables only support "
                    "scalar (1-D) spatial coordinates because the "
                    "spline grid scales as n_grid_x**(2d). Use lazy "
                    "mode (omit x_max / n_grid_x in "
                    "precompute_C_table_general) for d-dim inputs."
                )
            if self._c_general_axes is not None:
                xs = self._c_general_axes[2]
                remedy = (
                    "Rebuild the propagators with a larger x_max, or leave "
                    "x_max out for a lazy table."
                )
                _refuse_past_table("x1", x1_b, xs[0], xs[-1], remedy)
                _refuse_past_table("x2", x2_b, xs[0], xs[-1], remedy)
            pts = np.stack([t1, t2, x1_b, x2_b], axis=-1)
            return self._assemble_C(
                [itp(pts) for itp in self._c_general_interpolators], n
            )
        if self._lazy_general is not None:
            return self._lazy_lookup(
                self._lazy_general,
                t1,
                t2,
                x1_b,
                x2_b)
        raise RuntimeError(
            "homogeneity='general' but no general table has been "
            "built — call precompute_C_table_general() first"
        )

    def _closed_form_at_batch_diag(
        self,
        t1: np.ndarray,
        t2: np.ndarray,
        x1,
        x2,
    ) -> np.ndarray:
        """Direct lookup for ``closed_form_only=True``: skips every
        spline, calls ``self._C_value_direct`` (the user's c_fn), and
        returns either per-sample diagonals or full matrices.

        When ``model.diag_C`` is true the return shape is ``(n, N)``.
        Otherwise the full-matrix return shape is ``(n, N, N)``.

        Two contracts:

        * **Vectorised** (``self._closed_form_vectorized = True``):
          single call ``c_fn(x1, t1, x2, t2) -> (n, N, N)``.  Recommended
          for sweep performance.
        * **Per-sample** (default): Python loop calling
          ``c_fn(x1_i, t1_i, x2_i, t2_i) -> (N, N)`` per sample.
          Always correct, but ~50-100x slower than the vectorised
          path on typical workloads; only practical for small
          point evaluations or testing.
        """
        N = self.model.n_components
        n = len(t1)

        # Per-sample arrays, so the loop body can always slice ``x1_b[i]``.
        x1_b = _positions_per_sample(x1, n)
        x2_b = _positions_per_sample(x2, n)

        if self._closed_form_vectorized:
            full = np.asarray(_C_value_direct_on_copies(self, x1_b, t1, x2_b, t2))
            if full.ndim != 3 or full.shape[0] != n:
                raise ValueError(
                    f"vectorised closed-form c_fn must return shape "
                    f"(n, N, N); got {full.shape} for n={n}, N={N}."
                )
            # Owned copies, never the hook's array or a view of it: the
            # integrators keep several lookups alive at once (one per C
            # factor), so a hook that fills and returns one preallocated
            # buffer would otherwise turn every earlier factor into the
            # last one.  The per-sample path below and C_value's memo
            # copy too.
            if not self.model.diag_C:
                return full.astype(float, copy=True)
            # Diagonal per sample: result[i, a] = full[i, a, a].
            return np.einsum("iaa->ia", full).astype(float, copy=True)

        # Per-sample fallback. Slow but always correct.
        if self.model.diag_C:
            result = np.empty((n, N), dtype=float)
            for i in range(n):
                xi = x1_b[i]
                xj = x2_b[i]
                C_mat = np.asarray(
                    _C_value_direct_on_copies(self, xi, t1[i], xj, t2[i])
                )
                result[i] = np.diag(C_mat)
            return result

        result_full = np.empty((n, N, N), dtype=float)
        for i in range(n):
            xi = x1_b[i]
            xj = x2_b[i]
            result_full[i] = np.asarray(
                _C_value_direct_on_copies(self, xi, t1[i], xj, t2[i])
            )
        return result_full

    def _lazy_lookup(
        self,
        lazy: "_LazyTimeSplineCache",
        t1: np.ndarray,
        t2: np.ndarray,
        x1: np.ndarray,
        x2: np.ndarray,
    ) -> np.ndarray:
        """Look up C via per-parameter 2-D lazy splines, grouping
        samples by memoization key so each distinct parameter value
        triggers at most one spline build.  ``(n, N)`` under ``diag_C``,
        ``(n, N, N)`` otherwise (see :meth:`_assemble_C`)."""
        n = len(t1)
        columns = [np.empty(n) for _ in self._c_table_entries()]
        if n == 1:
            splines = lazy.get_splines(x1[0], x2[0])
            for e, col in enumerate(columns):
                col[0] = splines[e](t1, t2, grid=False)[0]
            return self._assemble_C(columns, n)
        _, firsts, inverse, counts = np.unique(
            np.column_stack((x1, x2)),
            axis=0,
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )
        order = np.argsort(np.asarray(inverse).reshape(-1), kind="stable")
        samples = np.split(order, np.cumsum(counts)[:-1])
        seen: dict = {}
        for k, at_pair in zip(firsts, samples):
            key = lazy._make_key(x1[k], x2[k])
            seen.setdefault(key, (k, []))[1].append(at_pair)
        for rep_k, parts in seen.values():
            splines = lazy.get_splines(x1[rep_k], x2[rep_k])
            sel = np.concatenate(parts)
            t1_sel = t1[sel]
            t2_sel = t2[sel]
            for e, col in enumerate(columns):
                col[sel] = splines[e](t1_sel, t2_sel, grid=False)
        return self._assemble_C(columns, n)


# ---------------------------------------------------------------------------
# Step 4: Contraction & Integration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DynamicCouplingPromise:
    """Defers coupling-tensor materialisation to QMC-sample time.

    When any entry in ``coupling_values`` is a callable (e.g. a
    spacetime-dependent non-local vertex such as demo2's
    ``κ^{(3)}(x₁,t₁; x₂,t₂; x₃,t₃)``), the usual pre-QMC
    :meth:`DiagramTerm.evaluate_coupling` path doesn't apply, since
    the callable's output changes with each sample's QMC time
    coordinates.  This class packages everything needed for the
    per-sample path:

    - the list of static (``ndarray``) symbol values,
    - the dynamic (callable) values, keyed by **leg order**: each
      distinct ``(name, spatial_args)`` occurrence of a callable symbol
      in the ``coupling_sum`` has its own key (``K@0``, ``K@1``, ...;
      see :func:`~sft_wick.perturbation._split_callable_occurrences`),
    - for each key, the ψ-leg spatial labels in that occurrence's leg
      order, so we can look up each leg's time and position per sample,
    - the parent :class:`DiagramTerm` with its ``coupling_sum``
      rewritten to those keys, so that every term reads the tensor
      evaluated at its own legs.

    The per-sample evaluator :meth:`evaluate_at` materialises the
    dynamic tensors using the sample's ``(times, positions)`` and
    then delegates to the static-path
    :meth:`DiagramTerm.evaluate_coupling` with a fully-numeric
    ``coupling_values`` dict, so the contraction code stays the
    same.
    """

    #: The parent :class:`DiagramTerm`, with each callable symbol in its
    #: ``coupling_sum`` renamed to its per-leg-order key (see
    #: :meth:`DiagramTerm.build_integrand`).
    diagram_term: Any

    #: Static coupling values: already materialised arrays.
    static_values: dict

    #: Dynamic coupling values: a mapping ``key -> callable(n_list,
    #: t_list)`` returning an ``ndarray``.  One callable appears under
    #: one key per leg order at which its symbol occurs.
    dynamic_values: dict

    #: Per-key tuple of ψ-leg spatial labels, in the leg order of that
    #: occurrence in the diagram's ``coupling_sum``.
    spatial_args_by_name: dict

    #: Component-index pins forwarded from ``build_integrand``.
    fixed_indices: dict

    def evaluate_at(
        self,
        times: dict,
        positions: dict,
    ) -> np.ndarray:
        """Materialise the dynamic callables at this sample's
        ``(times, positions)`` and return the per-sample
        ``coupling_array``.

        Args:
            times: ``{spatial_label: time_value}`` for all spatial
                labels referenced by any dynamic symbol's legs.
            positions: ``{spatial_label: position_value}`` same.
        """
        sample_cv = dict(self.static_values)
        for name, fn in self.dynamic_values.items():
            legs = self.spatial_args_by_name[name]
            t_list = np.array([times[s] for s in legs], dtype=float)
            n_list = np.array([positions[s] for s in legs], dtype=float)
            if getattr(fn, "vectorized", False):
                # A callable declared under the BATCHED contract expects
                # ``(m_legs, n_samples)`` and returns
                # ``(n_samples,) + kappa_shape``.  Calling it with the
                # per-sample ``(m_legs,)`` arrays instead would hand it the
                # wrong contract silently, and some such callables
                # broadcast and return a plausible wrong shape rather
                # than raising.  Run it
                # as a batch of one and unwrap.  ``np.array`` copies, here
                # and below: the tensors of every symbol stay alive until
                # the contraction, so a callable that fills and returns one
                # buffer would give every symbol the last one's values.
                stacked = np.array(fn(n_list[:, None], t_list[:, None]))
                if stacked.shape[0] != 1:
                    raise ValueError(
                        f"vectorized callable {name!r} returned shape "
                        f"{stacked.shape}; expected a leading axis of "
                        f"length 1 for a single-sample evaluation."
                    )
                sample_cv[name] = stacked[0]
            else:
                sample_cv[name] = np.array(fn(n_list, t_list))
        return self.diagram_term.evaluate_coupling(
            sample_cv,
            self.fixed_indices,
        )

    @cached_property
    def _scalar_probe_candidate(self):
        """Only immutable structure is cached; each numeric proof stays live."""
        from .perturbation import DiagramTerm
        from .expressions import Product, Sum

        dt = self.diagram_term
        if type(dt) is not DiagramTerm:
            return False
        expr = dt.coupling_sum
        return (
            type(expr) is Sum and type(expr.terms) is tuple and len(expr.terms) >= 4
        ) or (
            type(expr) is Product
            and type(expr.factors) is tuple
            and len(expr.factors) >= 4
        )

    def evaluate_at_batch(
        self,
        label_t: dict,
        label_x: dict,
        n_samples: int,
    ) -> np.ndarray:
        """Vectorised batch evaluator, returning the per-sample
        coupling as a complex array of shape ``(n_samples,) +
        prop_shape``, where ``prop_shape`` is the shape of the
        diagram's surviving :attr:`DiagramTerm.propagator_indices`
        (so ``(n_samples,)`` when the contraction is fully scalar).

        ``label_t`` and ``label_x`` map every spatial label appearing
        in any dynamic symbol's legs to a ``(n_samples,)`` time
        array / position respectively. Each ``label_x`` entry may be:

        * a scalar (the historical / 1-D translation case), which produces
          a per-leg broadcast of shape ``(n_samples,)``;
        * a ``(d,)`` vector (e.g. a 3-D unit vector under
          ``homogeneity='rotation'``), which produces a per-leg broadcast
          of shape ``(n_samples, d)``.

        The user callable then sees per-leg slices of shape ``(m,)`` or
        ``(m, d)`` respectively (or, in the vectorised contract below,
        ``(m, n_samples)`` / ``(m, n_samples, d)``). All legs of one
        symbol must have the same position shape; mixing scalar and
        vector legs raises from ``np.stack``.

        Per-symbol fast path: when the user marks a callable
        ``vectorized=True`` (e.g. via
        :class:`~sft_wick.workflow.specs.NonLocalVertex(coupling_vectorized=True)`
        or by setting ``fn.vectorized = True``), the wrapped
        callable receives ``(m_legs, n_samples)`` arrays (or
        ``(m_legs, n_samples, d)`` for d-dim positions) in a single
        call and returns a tensor of shape
        ``(n_samples,) + (N,)*order``. Otherwise the callable is
        called directly, once per sample: ``n_samples`` calls per leg
        order, whether or not the contraction falls back to the
        per-sample loop described next.

        Either way, the symbolic contraction itself runs once over
        the whole sample axis via
        :meth:`DiagramTerm.evaluate_coupling_batched`, falling back
        to a per-sample :meth:`DiagramTerm.evaluate_coupling` loop
        when the batched evaluator meets a node type it cannot
        handle.

        **Propagator-indexed output.**  When the contraction does not
        collapse to a scalar (a κ leg index survives onto a C
        propagator, as in demo2's order-4 F³κ³ diagrams), the
        returned array keeps those axes.  Callers must then contract
        it against the C-propagator product one index assignment at a
        time, exactly as the static branch does; see
        :meth:`DiagramIntegrand._dynamic_values`.  Before 0.4.0 this
        case raised ``NotImplementedError``; the static-vs-dynamic
        agreement that replaced it is locked by ``DC1`` in
        ``tests/test_dynamic_coupling.py``.
        """
        # Pre-build (m, n_samples) leg arrays once per symbol, so the
        # per-sample loop only does scalar slicing.
        per_symbol_legs: dict = {}
        for name in self.dynamic_values:
            legs = self.spatial_args_by_name[name]
            t_2d = np.stack(
                [np.asarray(label_t[s], dtype=float) for s in legs],
                axis=0,
            )  # shape (m, n_samples)
            leg_x_arrs = []
            for s in legs:
                x_val = np.asarray(label_x[s], dtype=float)
                # Positions per leg are *fixed* across samples (they
                # are the integration's external coordinates), so we
                # broadcast a leading sample axis of length n_samples.
                #
                # Two regimes:
                # * scalar position (x_val.ndim == 0): broadcast to
                #   (n_samples,), the historical case; produces a
                #   final n_arr of shape (m, n_samples) and a per-
                #   sample slice of shape (m,) for the user callable.
                # * d-dim vector position (x_val.ndim >= 1): broadcast
                #   to (n_samples, *x_val.shape); produces a final
                #   n_arr of shape (m, n_samples, *vec_shape) and a
                #   per-sample slice of shape (m, *vec_shape) for the
                #   user callable.  The user callable is responsible
                #   for handling its own per-leg position shape.
                if x_val.ndim == 0:
                    leg_x_arrs.append(np.full((n_samples,), float(x_val)))
                else:
                    leg_x_arrs.append(
                        np.broadcast_to(x_val[None, ...], (n_samples,) + x_val.shape)
                    )
            # Mixing scalar and vector legs (or different vector dims
            # across legs) makes ``np.stack`` raise, a clear shape
            # error that is more useful than a silent broadcast.
            n_arr = np.stack(leg_x_arrs, axis=0)
            # scalar legs -> (m, n_samples)
            # d-dim legs  -> (m, n_samples, *vec_shape)
            per_symbol_legs[name] = (n_arr, t_2d)

        # Build each dynamic symbol's per-sample tensors once, as a stack
        # of shape (n_samples, *kappa_shape): one call for a vectorised
        # callable, one call per sample for any other.  The sample-0
        # probe, the batched contraction and the per-sample fallback
        # loop below all read these stacks, so no sample is evaluated
        # twice.
        per_sample_tensors: dict = {}
        for name, fn in self.dynamic_values.items():
            n_arr, t_2d = per_symbol_legs[name]
            if getattr(fn, "vectorized", False):
                # A copy, as in :meth:`evaluate_at`: every stack stays alive
                # until the contraction.
                stacked = np.array(fn(n_arr, t_2d))
                if stacked.shape[0] != n_samples:
                    raise ValueError(
                        f"vectorized callable {name!r} returned shape "
                        f"{stacked.shape}; expected leading axis of "
                        f"length n_samples={n_samples}."
                    )
            else:
                sample0 = np.asarray(fn(n_arr[:, 0], t_2d[:, 0]))
                stacked = np.empty(
                    (n_samples,) + sample0.shape,
                    dtype=sample0.dtype,
                )
                stacked[0] = sample0
                for s in range(1, n_samples):
                    stacked[s] = np.asarray(fn(n_arr[:, s], t_2d[:, s]))
            per_sample_tensors[name] = stacked

        # Keep scalar-first errors unless native shape/index/arithmetic checks
        # prove sample zero succeeds. No tensor callbacks are repeated.
        sample_cv0 = dict(self.static_values)
        for name, stacked in per_sample_tensors.items():
            sample_cv0[name] = stacked[0]
        prop_shape = None
        if self._scalar_probe_candidate:
            from ._coupling_probe import known_scalar_shape

            prop_shape = known_scalar_shape(
                self.diagram_term,
                sample_cv0,
                self.fixed_indices,
                n_samples,
            )
        coup0 = None
        if prop_shape is None:
            coup0 = np.asarray(
                self.diagram_term.evaluate_coupling(
                    sample_cv0,
                    self.fixed_indices,
                )
            )
            prop_shape = coup0.shape
        out_shape = (n_samples,) + prop_shape

        # ------------------------------------------------------------
        # Vectorised fast path (default).
        # ------------------------------------------------------------
        # Contract the stacks with one call to
        # ``DiagramTerm.evaluate_coupling_batched``, replacing the
        # inner ``n_samples`` calls to ``_eval_symbolic`` with one
        # vectorised pass.
        #
        # If the symbolic ``coupling_sum`` contains a node type the
        # batched evaluator cannot handle (e.g. a ``Propagator`` or
        # ``IntegralOver`` slipped into a coupling expression), the
        # batched path raises ``NotImplementedError`` and we fall
        # back to the original per-sample loop below.  This mirrors
        # the safety net documented in
        # :func:`sft_wick.perturbation._eval_symbolic_batched`.
        batched_cv = {**self.static_values, **per_sample_tensors}
        try:
            couplings = self.diagram_term.evaluate_coupling_batched(
                batched_cv,
                n_samples=n_samples,
                fixed_indices=self.fixed_indices,
            )
            # ``evaluate_coupling_batched`` returns shape
            # ``(n_samples,) + prop_shape``.  Cross-check it against
            # the shape the sample-0 probe produced: a mismatch means
            # the batched evaluator disagreed with the scalar one, and
            # the scalar loop below is the trustworthy route.
            if couplings.shape != out_shape:
                raise NotImplementedError(
                    "evaluate_coupling_batched returned shape "
                    f"{couplings.shape}, expected {out_shape}; "
                    "falling back to the per-sample loop."
                )
            return np.ascontiguousarray(couplings).astype(complex, copy=False)
        except NotImplementedError:
            pass  # fall through to scalar per-sample loop below

        if coup0 is None:
            coup0 = np.asarray(
                self.diagram_term.evaluate_coupling(
                    sample_cv0,
                    self.fixed_indices,
                )
            )
            out_shape = (n_samples,) + coup0.shape
        couplings = np.empty(out_shape, dtype=complex)
        couplings[0] = coup0
        for s in range(1, n_samples):
            sample_cv = dict(self.static_values)
            for name, stacked in per_sample_tensors.items():
                sample_cv[name] = stacked[s]
            couplings[s] = np.asarray(
                self.diagram_term.evaluate_coupling(
                    sample_cv,
                    self.fixed_indices,
                )
            )
        return couplings


@dataclass(frozen=True)
class DiagramIntegrand:
    """A diagram's contribution decomposed for numerical integration.

    Combines coupling coefficients (Step 1) with spatial analysis (Step 2)
    for efficient numerical evaluation.

    The integrand has the factored form (when ``iso_R=True``)::

        R_product(times) × Σ_{prop_idx} coeff[idx] × Π_C C_values[idx]
    """

    #: The underlying DiagramTerm.
    diagram_term: "DiagramTerm"

    #: Spatial analysis result.
    spatial: SpatialStructure

    #: Coupling coefficient array from ``evaluate_coupling()``.
    #: When :attr:`dynamic_coupling` is set, this is a placeholder
    #: zero array and the real coupling value is computed per-sample
    #: in the integrator.
    coupling_array: np.ndarray

    #: Fixed component indices (e.g. ``{'a': 0, 'b': 1}``).
    #: Used by the QMC/GL integrators to resolve C-propagator
    #: component indices that are not summation variables.
    fixed_indices: dict[str, int] = field(default_factory=dict)

    #: Optional :class:`DynamicCouplingPromise` carrying per-sample
    #: coupling materialisation logic for spacetime-dependent
    #: (callable) coupling values.  ``None`` ⇒ the fully-static
    #: fast path is used (all coupling values were ndarrays).
    dynamic_coupling: Any = None

    # Query-owned execution choices; default folds preserve component order.
    _contraction_backend: str = "ordered"
    _contraction_max_bytes: int = DEFAULT_CONTRACTION_MAX_BYTES

    @property
    def _e_psi(self) -> int:
        """Number of external response legs of the observable (``E_psi``).

        ``getattr`` with a default keeps this working for
        :class:`~sft_wick.perturbation.DiagramTerm` objects deserialised from
        before the field existed.
        """
        return getattr(self.diagram_term, "n_external_response", 0)

    @cached_property
    def _coupling_legs(self) -> tuple[str, ...]:
        """The labels a callable coupling may read, sorted: every point of
        the diagram, and each leg that no propagator attaches to.

        Raises:
            ValueError: if a leg is no external point, no integration
                variable and no alias of one, so that it has no time.
        """
        spatial = self.spatial
        promise = self.dynamic_coupling
        legs = set(spatial.direction_map)
        # A custom promise implements ``evaluate_at_batch`` alone and sees
        # the points of the diagram.
        for name in getattr(promise, "dynamic_values", ()):
            legs.update(promise.spatial_args_by_name[name])
        legs = sorted(legs)
        timed = set(spatial.external_points) | set(spatial.time_integration_vars)
        timed |= {non_rep for non_rep, _rep in spatial.equal_time_aliases}
        unknown = [label for label in legs if label not in timed]
        if unknown:
            raise ValueError(
                f"callable coupling: the legs {unknown} have no time.  A leg "
                f"must be an external point of the diagram or one of its "
                f"integration variables."
            )
        return tuple(legs)

    def _leg_positions(
        self,
        positions: dict[str, Any] | None,
        default: Any,
        *,
        where: str,
    ) -> dict[str, Any]:
        """``{leg: position}`` for a callable coupling, the one rule for it.

        A leg sits at the coordinate of its direction group, which
        :meth:`_resolve_group_x` takes from ``positions``.  A leg of a group
        with no placed external point, and a leg that no propagator attaches
        to, sit at ``default`` (the ambient ``direction``).

        The rule does not depend on the cache.  A cache with no spatial table
        reads C by time, and that is a statement about C: the coupling is
        the user's function of position whatever the cache is.

        Raises:
            ValueError: as :meth:`_resolve_group_x`.
        """
        directions = self._resolve_group_x(
            self.spatial, positions, default, where=where
        )
        return self._legs_at(directions, default)

    def _legs_at(self, directions: dict[str, Any], default: Any) -> dict[str, Any]:
        """``{leg: position}`` from ``{direction_var: value}``."""
        direction_map = self.spatial.direction_map
        return {
            label: directions[direction_map[label]]
            if label in direction_map
            else default
            for label in self._coupling_legs
        }

    def _leg_times(self, time_of: Callable[[str], Any]) -> dict[str, Any]:
        """``{leg: time}``, one sample or one batch, from ``time_of(variable)``.

        A leg of an equal-time vertex and an R-absorbed leg read the variable
        of their representative, so ``time_of`` is never asked for an alias.
        """
        alias = dict(self.spatial.equal_time_aliases or ())
        return {
            label: time_of(alias.get(label, label)) for label in self._coupling_legs
        }

    def dynamic_coupling_array(
        self,
        times: dict[str, float],
        leg_positions: dict[str, Any],
    ) -> np.ndarray:
        """Materialise this sample's coupling tensor for a callable coupling.

        The scalar-loop counterpart of the batched backends' vectorised
        materialisation (:meth:`DynamicCouplingPromise.evaluate_at_batch`).
        It let a callable coupling be combined with a matrix-valued R while
        the batched backends still refused a matrix R.

        Args:
            times: ``{spatial_point: time}`` of this sample.
            leg_positions: ``{leg: position}`` from :meth:`_leg_positions`.
        """
        return self.dynamic_coupling.evaluate_at(
            self._leg_times(times.__getitem__), leg_positions
        )

    @cached_property
    def _r_index_mismatches(self) -> tuple[str | None, str | None]:
        """:func:`_r_index_mismatch` against a matrix R and a scalar R,
        computed once because :meth:`evaluate` consults it on every call."""
        dt = self.diagram_term
        return (_r_index_mismatch(dt, False), _r_index_mismatch(dt, True))

    def _require_r_indices_match(self, cache: Any) -> None:
        """Raise ``ValueError`` if this diagram's R-propagator indices
        contradict ``cache``'s R type (see :func:`_r_index_mismatch`)."""
        iso_R = _cache_r_type(cache)
        if iso_R is not None and self._r_index_mismatches[iso_R]:
            raise ValueError(self._r_index_mismatches[iso_R])

    @cached_property
    def _unresolved_labels(self) -> tuple[str, ...]:
        """Labels on this diagram's propagators that are neither a summation
        index, nor pinned by :attr:`fixed_indices`, nor a numeric literal."""
        known = {name for name, _ in self.diagram_term.propagator_indices}
        known.update(self.fixed_indices)
        legs = [leg for _, _, il, ir in self.spatial.c_propagators for leg in (il, ir)]
        for prop in self._kept_r_propagator_objects():
            legs += [prop.index_left, prop.index_right]
        return tuple(
            sorted(
                {
                    leg
                    for leg in legs
                    if leg and leg not in known and literal_component(leg) is None
                }
            )
        )

    def _require_labels_resolved(self) -> None:
        """Raise ``KeyError`` for a component label that names nothing, as the
        coupling resolvers do.  Such a propagator would be read as its trace,
        which is no component of the observable."""
        if self._unresolved_labels:
            raise KeyError(
                f"Component labels {list(self._unresolved_labels)} on the "
                f"propagators of this diagram are not summation indices or "
                f"literals.  Pass them via the fixed_indices argument of "
                f"build_integrand()."
            )

    def _c_position_pairs(self, directions: dict[str, Any]) -> list[tuple]:
        """``(x_l, x_r)`` of each C propagator, from ``directions`` in either
        spelling :meth:`evaluate` takes."""
        at = self.spatial.direction_map
        return [
            (
                directions.get(at[sp_l], directions.get(sp_l)),
                directions.get(at[sp_r], directions.get(sp_r)),
            )
            for sp_l, sp_r, _, _ in self.spatial.c_propagators
        ]

    def _require_c_sees_positions(
        self, cache: Any, positions: Any, direction: Any
    ) -> None:
        """:func:`_require_time_table_holds` at this diagram's positions."""
        if _time_only_table(cache):
            group_x = self._resolve_group_x(self.spatial, positions, direction)
            _require_time_table_holds(cache, self._c_position_pairs(group_x))

    def _admit(
        self, cache: Any, t_min: float | None, positions: Any, direction: Any
    ) -> float:
        """What every integrator checks first: :meth:`_admit_times` and the
        position refusals.  A position conflict is refused whether or not
        this diagram reads a position (:meth:`_resolve_group_x`)."""
        self._resolve_group_x(self.spatial, positions, direction)
        self._require_c_sees_positions(cache, positions, direction)
        return self._admit_times(cache, t_min)

    def _admit_times(self, cache: Any, t_min: float | None) -> float:
        """The R and label refusals, and the lower time limit
        (:func:`_integration_t_min`), which it returns."""
        self._require_r_indices_match(cache)
        self._require_labels_resolved()
        return _integration_t_min(cache, t_min)

    @cached_property
    def _coupling_kink_pairs(self) -> tuple[tuple[str, str], ...]:
        """:func:`_coupling_kink_candidates` of this integrand: the pairs of
        time arguments of its callable couplings that declare
        ``has_coincident_time_kinks``.  Empty for a static coupling."""
        return _coupling_kink_candidates(self.spatial, self.dynamic_coupling)

    def _c_kink_at(
        self, cache: Any, t_min: float, t_max: float, positions: Any, direction: Any
    ) -> bool:
        """:func:`_c_has_diagonal_kink` for this diagram: an undeclared
        closed-form C is probed on ``[t_min, t_max]``, where ``t_max`` is the
        time ceiling of the domain (:func:`_resolve_external_times`), so a
        pinned external past ``t_final`` is covered, at the positions the
        integrators read this diagram's C propagators at, which are the
        position of the table on a time-only table and the positions of the
        direction groups on any other cache, and so is a κ² that declares
        nothing.  No C propagator, no question."""
        spatial = self.spatial
        if not spatial.c_propagators:
            return False
        kernel = getattr(getattr(cache, "model", None), "kappa2", None)
        if not _c_kink_is_probed(cache) and (
            kernel is None or getattr(kernel, "has_diagonal_cusp", None) is not None
        ):
            return _c_has_diagonal_kink(cache)
        if _time_only_table(cache):
            at_table = getattr(cache, "_c_table_direction", direction)
            pairs = [(at_table, at_table)]
        else:
            group_x = self._resolve_group_x(spatial, positions, direction)
            pairs = self._c_position_pairs(group_x)
        return _c_has_diagonal_kink(cache, (t_min, t_max), tuple(pairs))

    def _split_externals(
        self, integrate_over: Any, lambda_f: float, external_times: dict | None
    ) -> tuple[list[str], dict[str, float], float]:
        """``(swept externals, {fixed external: its time}, time ceiling)``:
        :func:`_resolve_integrate_over` and :func:`_resolve_external_times`
        of this diagram."""
        spatial = self.spatial
        points = list(spatial.external_points)
        chosen = _resolve_integrate_over(integrate_over, points)
        swept = [v for v in points if v in chosen]
        fixed = [v for v in points if v not in chosen]
        times, ceiling = _resolve_external_times(
            spatial, fixed, lambda_f, external_times, swept
        )
        return swept, times, ceiling

    def _c_lookup(
        self, cache: Any, positions: Any, direction: Any, where: str
    ) -> Callable:
        """``(sp_l, sp_r, t_l, t_r) -> C batch``: at the positions of the two
        direction groups on a cache with a spatial table, by time on any
        other."""
        if not _cache_has_spatial_table(cache):
            return lambda _sp_l, _sp_r, t_l, t_r: _time_table_C_batch(cache, t_l, t_r)
        group_x = self._resolve_group_x(self.spatial, positions, direction, where=where)
        at = self.spatial.direction_map

        def lookup(sp_l: str, sp_r: str, t_l: np.ndarray, t_r: np.ndarray):
            return _C_at_fixed_positions(
                cache, t_l, t_r, group_x[at[sp_l]], group_x[at[sp_r]]
            )

        return lookup

    def _values_at(
        self,
        cache: Any,
        time_of: Callable[[str], np.ndarray],
        jacobians: np.ndarray,
        lookup_C: Callable,
        leg_x: dict[str, Any] | None,
        *,
        tag: str,
        couplings: Callable | None = None,
        reality: Any = None,
    ) -> np.ndarray:
        """The integrand at a batch of samples, zero where the Jacobian is.

        Args:
            time_of: ``label -> (n_samples,)`` times, resolving aliases.
            lookup_C: :meth:`_c_lookup`.  The lookup does not depend on the
                component assignment, so it is done once per propagator and
                the assignment loops only select components.
            leg_x: :meth:`_leg_positions` of a callable coupling.
            tag: names the method in the errors of a callable coupling.
            couplings: replaces ``evaluate_at_batch(label_t, label_x)`` of a
                callable coupling.
            reality: the accumulator of a chunked Gauss-Legendre grid.
        """
        n_samples = len(jacobians)
        r_product, r_batches = self._r_batches(cache, time_of, n_samples)
        c_batches = [
            (lookup_C(sp_l, sp_r, time_of(sp_l), time_of(sp_r)), il, ir)
            for sp_l, sp_r, il, ir in self.spatial.c_propagators
        ]
        if self.dynamic_coupling is None:
            values = self._static_values(
                c_batches, r_product, jacobians, r_batches=r_batches
            )
        else:
            label_t = self._leg_times(time_of)
            raw = (
                couplings(label_t, leg_x)
                if couplings is not None
                else self.dynamic_coupling.evaluate_at_batch(
                    label_t=label_t, label_x=leg_x, n_samples=n_samples
                )
            )
            # ``raw`` is (n_samples,) when the contraction is scalar, or
            # (n_samples,) + prop_shape when a κ leg index survives onto a
            # propagator; ``_dynamic_values`` handles both.
            values = self._dynamic_values(
                raw,
                c_batches,
                r_product,
                jacobians,
                where=tag,
                r_batches=r_batches,
                _reality=reality,
            )
        return np.where(jacobians > 0, values, 0.0)

    def _sum_kink_pieces(
        self,
        integrate_piece: Callable[[tuple, tuple], tuple[float, float]],
        c_kink: bool,
        time_orderings: list,
        fixed_times: dict,
        t_min: float,
        t_ceiling: float,
        lambda_f: float,
        swept: list[str],
        extra_orderings: tuple,
        extra_bounds: tuple,
    ) -> tuple[float, float] | None:
        """The summed ``(value, error)`` of the pieces the time domain splits
        into at its next kink, or ``None`` when no kink is left.

        A split at kink pairs comes first: ordering one pair can give a
        variable two unordered parents, so the pairs are re-detected under
        the orderings added so far, and each pass orders at least one more
        pair.  Then the kinks against a fixed external time, one cut per
        pass: the cut resolves that variable against that time and the
        candidates are a property of the diagram.  So the recursion through
        ``integrate_piece(orderings, cuts)`` terminates.
        """
        pairs = self._kink_pairs_under(c_kink, time_orderings, swept=swept)
        if pairs:
            pieces = [
                (tuple(extra_orderings) + extra, extra_bounds)
                for extra in _kink_orientations(
                    self.spatial, pairs, time_orderings, swept=swept
                )
            ]
        else:
            cuts = self._kink_cuts_under(
                c_kink,
                time_orderings,
                fixed_times,
                t_min,
                t_ceiling,
                extra_bounds,
                swept=swept,
                swept_ceiling=lambda_f,
            )
            if not cuts:
                return None
            var, t_star = cuts[0]
            pieces = [
                (extra_orderings, tuple(extra_bounds) + ((var, side, t_star),))
                for side in ("le", "ge")
            ]
        total, total_err = 0.0, 0.0
        for orderings, bounds in pieces:
            val, err = integrate_piece(orderings, bounds)
            total += val
            total_err += err
        return total, total_err

    def _kink_pairs_under(
        self, c_kink: bool, time_orderings, swept=()
    ) -> list[tuple[str, str]]:
        """:func:`_kink_pairs` of this integrand under ``time_orderings``:
        the pairs at which :meth:`integrate_moment_gauss_legendre` and
        :meth:`integrate_moment_nquad` split the time domain.  ``swept``
        names the externals ``integrate_over`` sweeps, which pair like an
        integration variable (:func:`_split_vars`)."""
        return _kink_pairs(
            self.spatial,
            c_kink,
            time_orderings,
            self._coupling_kink_pairs,
            swept=swept,
        )

    def _kink_cuts_under(
        self,
        c_kink: bool,
        time_orderings,
        fixed_times,
        t_min: float,
        t_ceiling: float,
        extra_bounds=(),
        swept=(),
        swept_ceiling=None,
    ) -> list[tuple[str, float]]:
        """:func:`_kink_constant_cuts` of this integrand: the ``(variable,
        time)`` cuts at which :meth:`integrate_moment_gauss_legendre` and
        :meth:`integrate_moment_nquad` split the time domain against an
        external point pinned at a fixed time. A swept external is a variable
        too and may cross a fixed time; its own upper limit is
        ``swept_ceiling`` (the integration ceiling)."""
        if not fixed_times:
            return []
        bounds = _constant_bounds(
            self.spatial,
            time_orderings,
            fixed_times,
            t_min,
            t_ceiling,
            extra_bounds,
            swept=swept,
            swept_ceiling=swept_ceiling,
        )
        return _kink_constant_cuts(
            self.spatial,
            c_kink,
            bounds,
            fixed_times,
            time_orderings,
            self._coupling_kink_pairs,
            extra_bounds,
            swept=swept,
        )

    def evaluate(
        self,
        times: dict[str, float],
        directions: dict[str, Any],
        cache: PropagatorCache,
        coupling_array: np.ndarray | None = None,
    ) -> complex:
        """Evaluate the integrand at specific time + direction coordinates.

        Args:
            times: ``{spatial_point: time_value}`` for ALL spatial points.
                Aliased (non-representative) legs of equal-time non-local
                vertices may be absent --- their values are filled in from
                the canonical representatives inside this method.  A
                cache's ``R_product`` that is not the package's own
                receives a copy (:func:`_r_product_times`), so a write
                into it changes neither the C lookups nor this dict.
            directions: ``{direction_var: value}`` for independent direction
                variables (one per R-connected component, keyed by
                the representative name from ``spatial.direction_map``).
            cache: A :class:`PropagatorCache` for propagator evaluation.
            coupling_array: this sample's coupling tensor, overriding the
                static :attr:`coupling_array`.  Required when the integrand
                carries a spacetime-dependent (callable) coupling, since the
                static attribute is then a zeros placeholder;
                :meth:`dynamic_coupling_array` produces it.

        Returns:
            Scalar value of the integrand (complex if coupling is complex).

        Raises:
            NotImplementedError: if this integrand carries a spacetime-dependent
                (callable) coupling and no ``coupling_array`` was supplied.
                It would otherwise read the zeros placeholder and return 0.
            ValueError: if this diagram's R-propagator component indices
                contradict ``cache``'s R type, as for
                :func:`integrate_diagrams`.  Also if ``directions`` is
                written in the point-keyed spelling and places two points
                of one direction group at different coordinates; see
                :func:`_require_one_position_per_group`.  Also if a time-only
                C table does not hold the C at ``directions``, as for the
                integrators; see :func:`_require_time_table_holds`.
        """
        if self.dynamic_coupling is not None and coupling_array is None:
            raise NotImplementedError(
                "DiagramIntegrand.evaluate() cannot evaluate a "
                "spacetime-dependent (callable) coupling without an explicit "
                "`coupling_array`: the static attribute is a zeros "
                "placeholder on the dynamic path, so the result would "
                "silently be 0.  Pass coupling_array="
                "integrand.dynamic_coupling_array(times, leg_positions), where "
                "leg_positions maps each leg of the coupling to its position, "
                "or use method='gauss_legendre' / method='qmc_vectorized'."
            )
        self._require_r_indices_match(cache)
        self._require_labels_resolved()
        dt = self.diagram_term
        spatial = self.spatial
        # ``directions`` is normally keyed by direction variable, one entry
        # per group, and cannot express two positions in one group.  The C
        # lookups below fall back to the POINT's own name when the direction
        # variable is absent (``directions.get(dir_l, directions.get(sp_l))``),
        # which can, so check the point-keyed spelling here.
        _require_one_position_per_group(
            spatial, directions, where="DiagramIntegrand.evaluate"
        )
        if _time_only_table(cache):
            _require_time_table_holds(cache, self._c_position_pairs(directions))
        coeff = self.coupling_array if coupling_array is None else coupling_array
        model = cache.model

        # An aliased leg reads its representative, as on the batched paths
        # (_batch_time_reader); the propagator lookups below (``cache.R_product``,
        # ``times[sp_l]`` etc.) then need no branch of their own.
        if spatial.equal_time_aliases:
            times = dict(times)
            for non_rep, rep in spatial.equal_time_aliases:
                if rep in times:
                    times[non_rep] = times[rep]

        # --- R product (scalar when iso_R) ---
        #
        # ``PropagatorCache.R_product`` is scalar-only.
        # For matrix-valued R, resolve component indices alongside the
        # C contraction below so order-0 R diagrams do not try to cast an
        # (N, N) matrix to float.  The C lookups below read ``times`` again,
        # so a user-written R_product receives a copy (_r_product_times).
        r_val = (
            cache.R_product(
                _kept_r_propagators(spatial), _r_product_times(cache, times)
            )
            if model.iso_R
            else 1.0
        )

        # --- Evaluate C propagators ---
        prop_idx = dt.propagator_indices
        if not prop_idx:
            # No propagator indices → scalar coupling, evaluate C without
            # summation, but still honour the integrand's
            # ``fixed_indices`` (observable component labels like
            # ``a``, ``b``).  Without this an unresolved leg contracts
            # to the trace (:func:`_select_C_batch`) and picks up a
            # spurious factor of N at order 0 for
            # ``⟨φ_a(x) φ_b(y)⟩``-style observables.
            c_val = self._scalar_c_product(
                1.0, times, directions, cache, self.fixed_indices
            )
            if not model.iso_R:
                c_val *= self._evaluate_r_product_general(
                    times,
                    cache,
                    self.fixed_indices,
                )
            return complex(r_val * complex(coeff) * c_val)

        # --- Map each C propagator to its propagator-index axis ---
        idx_names = [name for name, _ in prop_idx]

        if model.iso_R and model.diag_C:
            return self._evaluate_diag_fast(
                r_val,
                times,
                directions,
                cache,
                {name: ax for ax, name in enumerate(idx_names)},
                coupling_array=coeff,
            )
        else:
            return self._evaluate_general(
                r_val,
                times,
                directions,
                cache,
                idx_names,
                coupling_array=coeff,
            )

    def _evaluate_diag_fast(
        self,
        r_val: float,
        times: dict[str, float],
        directions: dict[str, Any],
        cache: PropagatorCache,
        idx_name_to_axis: dict[str, int],
        coupling_array: np.ndarray | None = None,
    ) -> complex:
        """Fast path for iso_R + diag_C case.

        Each C propagator contributes a diagonal vector; contraction is
        element-wise multiplication then summation.
        """
        spatial = self.spatial
        coeff = self.coupling_array if coupling_array is None else coupling_array
        n_axes = coeff.ndim

        contracted = coeff.copy()
        for sp_l, sp_r, il, ir in spatial.c_propagators:
            dir_l = spatial.direction_map[sp_l]
            dir_r = spatial.direction_map[sp_r]
            n_l, n_r = _c_lookup_positions(
                cache,
                "C_diagonal",
                directions.get(dir_l, directions.get(sp_l)),
                directions.get(dir_r, directions.get(sp_r)),
            )
            c_diag = cache.C_diagonal(n_l, times[sp_l], n_r, times[sp_r])

            if il is None and ir is None:
                # iso_C stripped both legs; see :func:`_isotropic_C`.
                contracted = contracted * _isotropic_C(c_diag)
                continue

            ax_l, val_l = self._diag_c_leg(il, idx_name_to_axis)
            ax_r, val_r = self._diag_c_leg(ir, idx_name_to_axis)

            # A leg with no label makes the propagator contract to its trace,
            # as in the ``a is None`` branch of :func:`_select_C_batch`.
            # A label that names nothing is refused
            # earlier (:meth:`_require_labels_resolved`).
            if (ax_l is None and val_l is None) or (ax_r is None and val_r is None):
                contracted = contracted * c_diag.sum()
                continue

            # ``diag_C`` ⇒ ``C_{ab} = δ_{ab} c_diag[a]``, so the two legs are
            # tied by a Kronecker delta.  ``apply_diagonal(diag_C=True)``
            # merges them into one name, but a term expanded WITHOUT diag_C
            # and evaluated against a diag_C cache still arrives with
            # ``il != ir``; using ``il`` alone then keeps a cross-component
            # term that every other backend sets to zero.
            shape = [1] * n_axes
            if ax_l is not None and ax_r is not None:
                if ax_l == ax_r:
                    shape[ax_l] = len(c_diag)
                    factor = c_diag.reshape(shape)
                else:
                    # δ across two distinct summation axes.  ``np.diag`` is
                    # symmetric, so the axis order of the reshape is
                    # immaterial.
                    shape[ax_l] = shape[ax_r] = len(c_diag)
                    factor = np.diag(c_diag).reshape(shape)
            elif ax_l is not None or ax_r is not None:
                # One summed leg, one pinned: δ collapses the sum to the
                # single assignment where the summation index equals the pin.
                axis = ax_l if ax_l is not None else ax_r
                pinned = val_r if ax_l is not None else val_l
                one_hot = np.zeros_like(c_diag)
                one_hot[pinned] = c_diag[pinned]
                shape[axis] = len(c_diag)
                factor = one_hot.reshape(shape)
            else:
                # Both legs pinned (observable labels like ``a``/``b``, or
                # 1-indexed literals).
                factor = c_diag[val_l] if val_l == val_r else 0.0
            contracted = contracted * factor

        total = contracted.sum()
        return complex(r_val * total)

    def _diag_c_leg(
        self,
        idx_name: str | None,
        idx_name_to_axis: dict[str, int],
    ) -> tuple[int | None, int | None]:
        """Classify one leg of a diagonal C propagator.

        Returns ``(axis, value)`` with at most one of them set:

        * ``(axis, None)``: the label is a propagator summation index and
          names an axis of the coupling array, to be contracted over.
        * ``(None, value)``: the label is pinned to a definite component,
          either by :attr:`fixed_indices` (an observable label such as
          ``a``) or as a 1-indexed literal.
        * ``(None, None)``: no component information; the caller falls back
          to the trace.

        Resolving against :attr:`fixed_indices` rather than an empty map is
        what keeps this path agreeing with :meth:`_evaluate_general` and the
        batched backends for observables like ``⟨φ_a(x) φ_b(y)⟩``.
        """
        if idx_name is None:
            return None, None
        axis = idx_name_to_axis.get(idx_name)
        if axis is not None:
            return axis, None
        return None, DiagramIntegrand._resolve_component(idx_name, self.fixed_indices)

    def _evaluate_general(
        self,
        r_val: float,
        times: dict[str, float],
        directions: dict[str, Any],
        cache: PropagatorCache,
        idx_names: list[str],
        coupling_array: np.ndarray | None = None,
    ) -> complex:
        """General path: explicit loop over propagator index combinations."""
        coeff = self.coupling_array if coupling_array is None else coupling_array
        dt = self.diagram_term
        prop_idx = dt.propagator_indices
        prop_shape = tuple(dim for _, dim in prop_idx)

        total = complex(0)
        for pidx in np.ndindex(*prop_shape):
            c_val = complex(coeff[pidx])
            if c_val == 0:
                continue

            # Merge the integrand's fixed component indices (e.g.
            # observable labels like ``a``, ``b``) into the
            # per-iteration summation ``idx_map``, matching the
            # vectorised path.  Without this merge, propagator legs
            # that reference observable labels contract to the trace
            # (:func:`_select_C_batch`, a spurious factor of N).
            idx_map = {
                **self.fixed_indices,
                **{name: val for name, val in zip(idx_names, pidx)},
            }

            c_val = self._scalar_c_product(c_val, times, directions, cache, idx_map)

            # For non-iso R: include R matrix elements
            if not cache.model.iso_R:
                c_val *= self._evaluate_r_product_general(times, cache, idx_map)

            total += c_val

        return r_val * total

    def _scalar_c_product(
        self,
        start: Any,
        times: dict[str, float],
        directions: dict[str, Any],
        cache: PropagatorCache,
        idx_map: dict[str, int],
    ) -> Any:
        """``start`` times the C propagators at one sample and one component
        assignment.  Selection is :func:`_select_C_batch`, on a batch of one,
        so it is the batched integrators' selection."""
        spatial = self.spatial
        value = start
        for sp_l, sp_r, il, ir in spatial.c_propagators:
            dir_l = spatial.direction_map[sp_l]
            dir_r = spatial.direction_map[sp_r]
            n_l, n_r = _c_lookup_positions(
                cache,
                "C_value",
                directions.get(dir_l, directions.get(sp_l)),
                directions.get(dir_r, directions.get(sp_r)),
            )
            C_mat = cache.C_value(n_l, times[sp_l], n_r, times[sp_r])
            if il is None and ir is None:
                value *= _isotropic_C(C_mat)
                continue
            a = self._resolve_component(il, idx_map)
            b = self._resolve_component(ir, idx_map)
            value *= _select_C_batch(np.asarray(C_mat)[np.newaxis], a, b)[0]
        return value

    def _evaluate_r_product_general(
        self,
        times: dict[str, float],
        cache: PropagatorCache,
        idx_map: dict[str, int],
    ) -> complex:
        """Evaluate matrix-valued R propagators for one component assignment."""
        result: complex = 1.0
        for r_prop in self._kept_r_propagator_objects():
            sl, sr = r_prop.spatial_left, r_prop.spatial_right
            if not _retarded(times[sl], times[sr]):
                return complex(0.0)
            R_mat = np.asarray(cache.R_time(times[sl], times[sr]))
            entry = self._r_entry(r_prop.index_left, r_prop.index_right, idx_map)
            result *= complex(np.trace(R_mat) if entry is None else R_mat[entry])
        return result

    @staticmethod
    def _r_entry(
        il: str | None, ir: str | None, idx_map: dict[str, int]
    ) -> tuple[int, int] | None:
        """The entry a matrix R propagator selects at one component
        assignment, or ``None`` when a leg is unresolved and it contributes
        its trace."""
        if not (il and ir):
            return None
        a = DiagramIntegrand._resolve_component(il, idx_map)
        b = DiagramIntegrand._resolve_component(ir, idx_map)
        return None if a is None or b is None else (a, b)

    def _kept_r_propagator_objects(self) -> tuple[Propagator, ...]:
        """This diagram's R :class:`Propagator` objects, minus absorbed pairs.

        The object-level counterpart of :func:`_kept_r_propagators`, which
        yields only ``(spatial_left, spatial_right)`` endpoint pairs.

        Endpoints do not identify an R propagator: two of them can join the
        *same* two points while carrying different component indices --- a
        local vertex with two ψ legs produces exactly that, e.g.
        ``R_{i1 i3}(y_0, y_1) R_{i2 i4}(y_0, y_1)``.  Resolving indices by
        endpoint lookup returns the first match for both factors and
        silently evaluates ``R[j,l] * R[j,l]`` instead of
        ``R[j,l] * R[k,m]``, so the matrix-R product loop must walk the
        propagator objects and read each factor's own indices.

        :func:`analyze_spatial` builds ``spatial.r_propagators`` by walking
        ``diagram_term.propagators`` in order, so this sequence matches it
        element for element.  Absorbed propagators are filtered by
        endpoint, the same rule :func:`_kept_r_propagators` applies.
        """
        absorbed = set(self.spatial.r_absorbed_pairs)
        return tuple(
            p
            for p in self.diagram_term.propagators
            if p.kind == "R" and (p.spatial_left, p.spatial_right) not in absorbed
        )

    def _r_batches(
        self,
        cache: Any,
        times: Callable[[str], np.ndarray],
        n_samples: int,
    ) -> tuple[np.ndarray, list]:
        """The R factors of the batched integrators, ``(r_product,
        r_batches)``.

        A scalar R (``cache.model.iso_R``) carries no component index, so
        its factors multiply into the ``(n_samples,)`` ``r_product`` once.
        A matrix R's entry depends on the component assignment: each kept R
        propagator becomes one entry of ``r_batches``
        (:meth:`_matrix_r_batches`), ``r_product`` stays 1, and
        :meth:`_r_factor` selects the entries per assignment.  Absorbed R
        propagators are skipped either way.

        Args:
            times: ``label -> (n_samples,)`` time array, resolving
                equal-time aliases.
        """
        r_product = np.ones(n_samples)
        if not cache.model.iso_R:
            return r_product, self._matrix_r_batches(cache, times)
        for sl, sr in _kept_r_propagators(self.spatial):
            r_product *= cache.R_time_batch(times(sl), times(sr))
        return r_product, []

    def _matrix_r_batches(
        self,
        cache: Any,
        times: Callable[[str], np.ndarray],
    ) -> list[tuple[np.ndarray, str | None, str | None, np.ndarray]]:
        """``(R_batch, index_left, index_right, nonzero)`` per kept R
        propagator of a matrix-valued R.

        ``R_batch`` is ``cache.R_matrix_batch`` at the propagator's two
        times, ``(n_samples, N, N)``; ``nonzero[a, b]`` records whether
        entry ``(a, b)`` is non-zero at any sample, so an assignment that
        reads an entry that is zero everywhere (an off-diagonal entry of a
        diagonal R) is skipped without touching the sample axis.  The
        propagator objects come from :meth:`_kept_r_propagator_objects`, so
        two R propagators between the same two points keep their own
        indices.
        """
        out = []
        for p in self._kept_r_propagator_objects():
            # Kept while the next R is looked up: owned (see _owned_batch).
            R_b = _owned_batch(
                cache,
                "R_matrix_batch",
                cache.R_matrix_batch(times(p.spatial_left), times(p.spatial_right)),
            )
            out.append((R_b, p.index_left, p.index_right, np.any(R_b != 0.0, axis=0)))
        return out

    @staticmethod
    def _r_factor(
        r_product: np.ndarray,
        r_batches: list,
        idx_map: dict[str, int],
        *,
        _trace_memo: dict | None = None,
    ) -> np.ndarray | None:
        """``r_product`` times each matrix R factor at one component
        assignment, or ``None`` when one of those factors is zero at every
        sample.

        Resolves indices the way :meth:`_evaluate_r_product_general` does:
        entry ``R[:, a, b]`` when both legs resolve, the trace otherwise.
        With no matrix factors (a scalar R) ``r_product`` is returned as it
        is.
        """
        out = r_product
        for R_b, il, ir, nonzero in r_batches:
            entry = DiagramIntegrand._r_entry(il, ir, idx_map)
            if entry is not None:
                if not nonzero[entry]:
                    return None
                out = out * R_b[(slice(None), *entry)]
                continue
            cached = None if _trace_memo is None else _trace_memo.get(id(R_b))
            if cached is None:
                trace = np.einsum("iaa->i", R_b)
                cached = (trace, bool(np.any(trace)))
                if _trace_memo is not None:
                    _trace_memo[id(R_b)] = cached
            trace, nonzero_trace = cached
            if not nonzero_trace:
                return None
            out = out * trace
        return out

    @staticmethod
    def _c_product(
        c_batches: "list[tuple[np.ndarray, str | None, str | None]]",
        idx_map: dict[str, int],
        n_samples: int,
        *,
        _reduction_memo: dict | None = None,
    ) -> np.ndarray:
        """Product of the C propagators at one component assignment, from
        batches whose lookup is already done."""
        cp = np.ones(n_samples)
        for C_batch, il, ir in c_batches:
            iso = il is None and ir is None
            a = None if iso else DiagramIntegrand._resolve_component(il, idx_map)
            b = None if iso else DiagramIntegrand._resolve_component(ir, idx_map)
            # A reduction that sums a batch recurs across the assignments.
            key = None
            if _reduction_memo is not None and (a is None or b is None):
                key = (id(C_batch), iso)
            factor = None if key is None else _reduction_memo.get(key)
            if factor is None:
                if iso:
                    factor = _isotropic_C(C_batch, batched=True)
                else:
                    factor = _select_C_batch(C_batch, a, b)
                if key is not None:
                    _reduction_memo[key] = factor
            cp = cp * factor
        return cp

    def _batch_reduction_memos(
        self,
        couplings: np.ndarray,
        c_batches: list,
        r_batches: list,
        r_product: np.ndarray,
        jacobians: np.ndarray | None,
        prop_idx: tuple,
    ) -> tuple[dict | None, dict | None]:
        """Empty, invocation-local reduction memos, or the original path.

        No numeric reduction or support scan happens here. Ordinary labels
        establish whether a reduction can recur; the original loop still
        resolves every assignment and validates only the factors it reaches.
        Subclasses/object arrays and custom index metadata can mutate borrowed
        buffers while being inspected, so retain their uncached behavior.
        """
        if (
            type(self) is not DiagramIntegrand
            or not prop_idx
            or type(self.fixed_indices) is not dict
        ):
            return None, None
        # A cheap, pure structural rejection precedes the full cache guards.
        # It neither reads numerical arrays nor invokes custom metadata.
        if (
            type(prop_idx) is not tuple
            or any(
                type(item) is not tuple or len(item) != 2 or type(item[0]) is not str
                for item in prop_idx
            )
            or any(type(label) is not str for label in self.fixed_indices)
        ):
            return None, None
        names = {name for name, _ in prop_idx}
        fi = self.fixed_indices

        def resolved(label):
            if type(label) is not str:
                return False
            if label in names:
                return True
            if label in fi:
                return type(fi[label]) is int or type(fi[label]) is bool
            return is_literal_component(label)

        if all(resolved(il) and resolved(ir) for _, il, ir in c_batches) and all(
            resolved(il) and resolved(ir) for _, il, ir, _ in r_batches
        ):
            return None, None
        for name, native in DiagramIntegrand._batch_reduction_helpers:
            if (
                name in self.__dict__
                or DiagramIntegrand.__dict__.get(name) is not native
            ):
                return None, None
        index_types = (
            int,
            bool,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        )
        if (
            type(prop_idx) is not tuple
            or any(
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or type(item[1]) not in index_types
                for item in prop_idx
            )
            or not any(dim > 1 for _, dim in prop_idx)
        ):
            return None, None
        fi = self.fixed_indices
        labels = [name for name, _ in prop_idx]
        labels.extend(fi)
        labels.extend(label for _, il, ir in c_batches for label in (il, ir))
        labels.extend(label for _, il, ir, _ in r_batches for label in (il, ir))
        if any(label is not None and type(label) is not str for label in labels) or any(
            value is not None and type(value) not in index_types
            for value in fi.values()
        ):
            return None, None
        idx = {**fi, **{name: 0 for name, _ in prop_idx}}

        def unresolved(label):
            return self._resolve_component(label, idx) is None

        has_c = any(
            (il is None and ir is None) or unresolved(il) or unresolved(ir)
            for _, il, ir in c_batches
        )
        has_r = any(
            not (il and ir) or unresolved(il) or unresolved(ir)
            for _, il, ir, _ in r_batches
        )
        if not (has_c or has_r):
            return None, None
        from .perturbation import DiagramTerm

        if (
            type(self.diagram_term) not in (DiagramTerm, SimpleNamespace)
            or type(self._e_psi) not in index_types
        ):
            return None, None
        arrays = [couplings, r_product]
        if jacobians is not None:
            arrays.append(jacobians)
        arrays.extend(c for c, _, _ in c_batches)
        arrays.extend(array for r, _, _, nz in r_batches for array in (r, nz))
        if any(
            type(array) is not np.ndarray or array.dtype.kind not in "biufc"
            for array in arrays
        ):
            return None, None
        return ({} if has_r else None), ({} if has_c else None)

    def _static_values(
        self,
        c_batches: "list[tuple[np.ndarray, str | None, str | None]]",
        r_product: np.ndarray,
        jacobians: np.ndarray,
        *,
        r_batches: list = (),
        where: str = " (coupling)",
    ) -> np.ndarray:
        """The integrand of a static coupling at every sample: the sum over
        the diagram's component assignments of coupling x R factors x C
        factors, times ``jacobians``.

        The static counterpart of :meth:`_dynamic_values`, shared by the
        QMC-vectorised and Gauss-Legendre integrators and
        :func:`integrate_two_point_qmc`.  ``c_batches`` holds the C lookups,
        ``r_product`` the scalar-R product and ``r_batches`` the matrix-R
        factors (see :meth:`_r_batches`).
        """
        coeff = self.coupling_array
        n_samples = int(r_product.shape[0])
        fi = self.fixed_indices
        prop_idx = self.diagram_term.propagator_indices
        if not prop_idx:
            c_val = _real_or_raise(coeff, self._e_psi, where=where)
            r_fac = self._r_factor(r_product, r_batches, fi)
            if r_fac is None:
                return np.zeros(n_samples)
            return r_fac * c_val * self._c_product(c_batches, fi, n_samples) * jacobians

        if self._contraction_backend == "einsum" and _native_dense_integrand(self):
            dense = try_static_contraction(
                coeff,
                prop_idx,
                fi,
                c_batches,
                r_product,
                jacobians,
                r_batches,
                self._e_psi,
                self._contraction_max_bytes,
            )
            if dense is not None:
                return dense

        memos = None, None
        if type(coeff) is not np.ndarray or coeff.ndim == 0 or coeff.size > 1:
            memos = self._batch_reduction_memos(
                coeff,
                c_batches,
                r_batches,
                r_product,
                jacobians,
                prop_idx,
            )

        def coefficient(pidx: tuple) -> Any:
            c_raw = coeff[pidx] if coeff.ndim > 0 else coeff
            # Only exact zeros are sparse: a small coefficient can multiply
            # a large kernel, so an absolute cutoff changes the observable.
            if c_raw == 0:
                return None
            return _real_or_raise(c_raw, self._e_psi, where=where)

        return self._fold_assignments(
            coefficient, c_batches, r_product, r_batches, memos, jacobians
        )

    def _fold_assignments(
        self,
        coefficient: Callable[[tuple], Any],
        c_batches: list,
        r_product: np.ndarray,
        r_batches: list,
        memos: tuple[dict | None, dict | None],
        jacobians: np.ndarray | None = None,
    ) -> np.ndarray:
        """The sum over the diagram's component assignments of
        ``coefficient(assignment)`` x R factors x C factors x ``jacobians``.
        ``coefficient`` returns ``None`` for an assignment that contributes
        nothing.  ``memos`` is :meth:`_batch_reduction_memos`.
        """
        n_samples = int(r_product.shape[0])
        fi = self.fixed_indices
        prop_idx = self.diagram_term.propagator_indices
        idx_names = [name for name, _ in prop_idx]
        prop_shape = tuple(dim for _, dim in prop_idx)
        r_memo, c_memo = memos
        r_keyword = {} if r_memo is None else {"_trace_memo": r_memo}
        c_keyword = {} if c_memo is None else {"_reduction_memo": c_memo}
        values = np.zeros(n_samples)
        for pidx in np.ndindex(*prop_shape):
            c_val = coefficient(pidx)
            if c_val is None:
                continue
            idx_map = {**fi, **dict(zip(idx_names, pidx))}
            r_fac = self._r_factor(r_product, r_batches, idx_map, **r_keyword)
            if r_fac is None:
                continue
            term = (
                c_val
                * r_fac
                * self._c_product(c_batches, idx_map, n_samples, **c_keyword)
            )
            values += term if jacobians is None else term * jacobians
        return values

    def _dynamic_values(
        self,
        couplings: np.ndarray,
        c_batches: "list[tuple[np.ndarray, str | None, str | None]]",
        r_product: np.ndarray,
        jacobians: np.ndarray | None = None,
        *,
        where: str = "",
        r_batches: list = (),
        _reality: _BatchRealityAccumulator | None = None,
    ) -> np.ndarray:
        """Combine a per-sample dynamic coupling with the C-propagator
        product: the dynamic counterpart of :meth:`_static_values`.

        ``couplings`` comes from
        :meth:`DynamicCouplingPromise.evaluate_at_batch` and is either

        * ``(n_samples,)``: the contraction collapsed to a scalar, so
          the C-propagator components are fixed by
          :attr:`fixed_indices` alone and one C-product suffices; or
        * ``(n_samples,) + prop_shape``: a κ leg index survived onto
          a C propagator (demo2's order-4 F³κ³), or onto a matrix-valued
          R.  Then the C-product depends on the index assignment, so we
          sum over ``np.ndindex(prop_shape)`` and rebuild it per
          assignment, mirroring the static prop-indexed branch exactly.

        ``c_batches`` holds ``(C_batch, index_left, index_right)`` per
        C propagator with the batched propagator lookup already done:
        the lookup does not depend on the component assignment, so
        hoisting it keeps the per-index loop to component selection.
        ``r_batches`` holds the matrix-R factors (see :meth:`_r_batches`),
        selected per assignment the same way.

        Returns the ``(n_samples,)`` real integrand, multiplied by
        ``jacobians`` when given.
        """
        n_samples = int(r_product.shape[0])
        fi = self.fixed_indices

        couplings = np.asarray(couplings)

        def real_values(raw, key):
            if _reality is not None:
                return _reality.project(raw, self._e_psi, where, key)
            return _real_batch_or_raise(raw, self._e_psi, where=where)

        if couplings.ndim <= 1:
            c_val = real_values(couplings, ())
            r_fac = self._r_factor(r_product, r_batches, fi)
            values = (
                np.zeros(n_samples)
                if r_fac is None
                else c_val * r_fac * self._c_product(c_batches, fi, n_samples)
            )
        else:
            prop_idx = self.diagram_term.propagator_indices
            prop_shape = tuple(dim for _, dim in prop_idx)
            if couplings.shape[1:] != prop_shape:
                raise ValueError(
                    f"dynamic coupling returned per-sample shape "
                    f"{couplings.shape[1:]}, but this diagram's "
                    f"propagator indices are {prop_idx} (shape "
                    f"{prop_shape}).  A callable κ must return the "
                    f"bare tensor over its OWN legs; the contraction "
                    f"onto propagator indices is done here."
                )
            native_projector = _reality is None
            if type(_reality) is _BatchRealityAccumulator:
                project = _reality.project
                native_projector = (
                    type(project) is MethodType
                    and project.__func__ is _BATCH_REALITY_PROJECT
                    and project.__self__ is _reality
                    and type(_reality._stats) is dict
                )
            memos = None, None
            if native_projector and couplings.size > n_samples:
                memos = self._batch_reduction_memos(
                    couplings,
                    c_batches,
                    r_batches,
                    r_product,
                    jacobians,
                    prop_idx,
                )

            def coefficient(pidx: tuple) -> Any:
                c_raw = couplings[(slice(None),) + pidx]
                return real_values(c_raw, pidx) if np.any(c_raw != 0) else None

            values = self._fold_assignments(
                coefficient, c_batches, r_product, r_batches, memos
            )

        if jacobians is not None:
            values = values * jacobians
        return values

    @staticmethod
    def _resolve_component(idx_name: str | None, idx_map: dict[str, int]) -> int | None:
        """Resolve a component index name to a 0-indexed integer."""
        return None if idx_name is None else pinned_component(idx_name, idx_map)

    _batch_reduction_helpers = (
        ("_r_factor", _r_factor),
        ("_c_product", _c_product),
        ("_resolve_component", _resolve_component),
    )

    @staticmethod
    def _resolve_group_x(
        spatial: "SpatialStructure",
        positions: dict[str, Any] | None,
        default: Any,
        *,
        where: str = "this evaluation",
    ) -> dict[str, Any]:
        """Map each direction group (by its direction-variable name) to
        a single spatial coordinate.

        Priority for each group:
        1.  If the group contains an external point whose position the
            caller specified in ``positions``, use that.
        2.  Otherwise fall back to ``default`` (typically 0 for
            integration-only groups, or whatever ``direction`` kwarg
            supplies for backward compat).

        This is what flows into ``cache.C_at_batch`` as the endpoint
        x-value when the cache has a spatial table built (of any
        homogeneity kind).

        ``positions`` values may be scalars (1-D / legacy) or
        arbitrary-dimensional vectors. The ``default`` likewise may
        be a scalar or a vector. The returned dict preserves whatever
        shape the user passed.  Downstream
        ``PropagatorCache.C_at_batch`` and the spatial Kappa2
        wrappers (``_SeparableTranslationKappa2``, ``_rotation_cos``,
        ...) all accept either form.

        Raises:
            ValueError: if ``positions`` places two points of one group at
                different coordinates.  Step 1 would then take whichever
                the ``frozenset`` yielded first, and that order changes
                with the labels and with ``PYTHONHASHSEED``; see
                :func:`_require_one_position_per_group`.
        """
        positions = _external_positions(spatial, positions)
        _require_one_position_per_group(spatial, positions, where=where)
        group_x: dict[str, Any] = {}
        for group in spatial.direction_groups:
            dvar_sample = spatial.direction_map[next(iter(group))]
            x_val: Any = default
            for p in group:
                if p in positions:
                    x_val = positions[p]
                    break
            group_x[dvar_sample] = x_val
        return group_x

    def _evaluate_zero_dimensional(
        self,
        lambda_f: float,
        cache: PropagatorCache,
        *,
        direction: Any = 0,
        positions: dict[str, Any] | None = None,
        external_times: dict[str, float] | None = None,
    ) -> float:
        """Evaluate an integrand with no surviving time variables.

        Dynamic non-local couplings still need to be materialised here. This
        occurs for already-R-contracted vertices whose absorbed R legs alias
        directly onto fixed external points.
        """
        spatial = self.spatial
        _where = "DiagramIntegrand._evaluate_zero_dimensional"
        directions = self._resolve_group_x(spatial, positions, direction, where=_where)

        fixed_times, _ceiling = _resolve_external_times(
            spatial,
            spatial.external_points,
            lambda_f,
            external_times,
        )
        if self.dynamic_coupling is None:
            val = self.evaluate(fixed_times, directions, cache)
            return float(_real_or_raise(val, self._e_psi, where=" (zero-dimensional)"))

        time_of = _batch_time_reader(
            np.empty((1, 0)), {}, fixed_times, lambda_f, spatial.equal_time_aliases
        )
        values = self._values_at(
            cache,
            time_of,
            np.ones(1),
            self._c_lookup(cache, positions, direction, _where),
            self._leg_positions(positions, direction, where=_where),
            tag=" (dynamic coupling, zero-dimensional)",
        )
        return float(values[0])

    def make_scipy_integrand(
        self,
        external_times: dict[str, float],
        external_directions: dict[str, Any],
        cache: PropagatorCache,
    ) -> Callable:
        """Return a callable for ``scipy.integrate.nquad``.

        The returned function accepts the integration time variables as
        positional arguments in the order of
        ``spatial.time_integration_vars`` and returns the integrand value.
        External times and directions are baked in.

        Args:
            external_times: ``{point_name: time}`` for external spatial points.
            external_directions: ``{direction_var: value}`` for all independent
                direction variables, or the point-keyed spelling of
                :meth:`evaluate`.
            cache: A :class:`PropagatorCache`.

        Returns:
            Callable ``f(*time_args) → float``.  A spacetime-dependent
            (callable) coupling is evaluated at every call, as in
            :meth:`integrate_moment_nquad`.
        """
        spatial = self.spatial
        int_vars = spatial.time_integration_vars
        dyn = self.dynamic_coupling is not None
        at = spatial.direction_map
        groups = {at[p]: x for p, x in external_directions.items() if p in at}
        leg_x = self._legs_at({**groups, **external_directions}, 0.0) if dyn else None

        def integrand(*time_args: float) -> float:
            times = dict(external_times)
            for var, val in zip(int_vars, time_args):
                times[var] = val
            ca = self.dynamic_coupling_array(times, leg_x) if dyn else None
            result = self.evaluate(times, external_directions, cache, coupling_array=ca)
            return _real_or_raise(result, self._e_psi, where=" (make_scipy_integrand)")

        return integrand

    def integration_bounds(
        self,
        external_times: dict[str, float],
        t_min: float,
    ) -> list:
        """Return integration bounds for ``scipy.integrate.nquad``.

        Time ordering constraints from R causality translate to
        variable-dependent bounds.  For ``nquad``, bounds can be callables.

        The variables are ordered as ``spatial.time_integration_vars``,
        which :func:`_topological_sort_times` emits **earliest time first**,
        so ``int_vars[0]`` is scipy's *innermost* integral.

        The domain is :func:`_causal_bounds` with the latest external time as
        ceiling, and the ranges are :func:`_nquad_ranges`, as in
        :meth:`integrate_moment_nquad`.

        Args:
            external_times: Time of each external point.
            t_min: The lower time limit, ``t_min`` of the propagator model.
                It has no default: this method holds no cache to read it from.

        Returns:
            List of bounds, one per integration variable.  Each is either
            ``(lo, hi)`` or a callable for dependent bounds.
        """
        int_vars = list(self.spatial.time_integration_vars)
        ceiling = max(external_times.values()) if external_times else 1.0
        bounds = _causal_bounds(
            self.spatial, int_vars, (), external_times, ceiling, ceiling, t_min
        )
        return _nquad_ranges(bounds, int_vars)

    def to_latex(self) -> str:
        r"""Render the decomposed integrand with explicit coordinates.

        Shows direction identifications, time-ordered integrals,
        factored R product, and the component sum over C propagators.
        """
        from .expressions import _latex_index

        spatial = self.spatial
        dt = self.diagram_term
        parts: list[str] = []

        # --- Direction identification note ---
        for group in spatial.direction_groups:
            if len(group) > 1:
                rep = None
                for p in sorted(group):
                    if p in spatial.external_points:
                        rep = p
                        break
                if rep is None:
                    rep = sorted(group)[0]
                others = sorted(group - {rep})
                if others:
                    parts.append(
                        rf"\text{{[all directions}} = \hat{{n}}_{{{_latex_index(rep)}}}]"
                    )

        # --- Time-ordered integrals ---
        body_parts: list[str] = []

        # R product
        r_strs = []
        for sl, sr in _kept_r_propagators(spatial):
            r_strs.append(rf"R(t_{{{_latex_index(sl)}}}, t_{{{_latex_index(sr)}}})")
        if r_strs:
            body_parts.append(" ".join(r_strs))

        # Summation + C factors
        prop_idx = dt.propagator_indices
        sum_prefix = ""
        for name, dim in prop_idx:
            sum_prefix += rf"\sum_{{{_latex_index(name)}=1}}^{{{dim}}} "

        c_strs = []
        for sp_l, sp_r, il, ir in spatial.c_propagators:
            dir_l = spatial.direction_map[sp_l]
            dir_r = spatial.direction_map[sp_r]
            idx_str = ""
            if il is not None and ir is not None:
                idx_str = f"_{{{_latex_index(il)}{_latex_index(ir)}}}"
            c_strs.append(
                rf"C{idx_str}({dir_l}, t_{{{_latex_index(sp_l)}}};\, "
                rf"{dir_r}, t_{{{_latex_index(sp_r)}}})"
            )

        coeff_str = r"\mathrm{coeff}"
        if prop_idx:
            idx_sub = ",".join(_latex_index(n) for n, _ in prop_idx)
            coeff_str = rf"\mathrm{{coeff}}_{{{idx_sub}}}"

        inner = rf"{sum_prefix}{coeff_str} " + " ".join(c_strs)
        body_parts.append(inner)

        body = r" \cdot ".join(body_parts)

        # Wrap with time integrals
        for var in reversed(list(spatial.time_integration_vars)):
            body = rf"\int \mathrm{{d}}t_{{{_latex_index(var)}}}\, {body}"

        result = body
        if parts:
            result = " ".join(parts) + r" \quad " + body

        return result

    def integrate_moment_qmc(
        self,
        lambda_f: float,
        cache: PropagatorCache,
        t_min: float | None = None,
        direction: Any = 0,
        n_samples: int = 2**14,
        seed: int | None = None,
        positions: dict[str, float] | None = None,
        integrate_over: Any = None,
        external_times: dict[str, float] | None = None,
    ) -> tuple[float, float]:
        """Integrate over all time variables using Quasi-Monte Carlo.

        Scalar-loop sibling of
        :meth:`integrate_moment_qmc_vectorized`; see that method's
        docstring for the ``integrate_over`` kwarg semantics.
        """
        from scipy.stats import qmc

        t_min = self._admit(cache, t_min, positions, direction)

        # A spacetime-dependent (callable) coupling used to be refused here,
        # which left the combination "callable coupling + MATRIX-valued R"
        # computable by no backend at all: this loop rejected the callable,
        # and ``qmc_vectorized`` / ``gauss_legendre`` rejected matrix R.  The
        # scalar loop is in fact the natural home for the per-sample callable
        # contract (it already visits one sample at a time), so it now
        # materialises the coupling per sample instead.
        dyn = self.dynamic_coupling is not None

        spatial = self.spatial
        int_vars_pf = list(reversed(spatial.time_integration_vars))
        ext_integrated, fixed_times, t_ceiling = self._split_externals(
            integrate_over, lambda_f, external_times
        )
        sobol_vars = ext_integrated + int_vars_pf
        n_total = len(sobol_vars)

        directions = self._resolve_group_x(
            spatial,
            positions,
            direction,
            where="DiagramIntegrand.integrate_moment_qmc",
        )
        leg_x = (
            self._leg_positions(
                positions, direction, where="DiagramIntegrand.integrate_moment_qmc"
            )
            if dyn
            else None
        )

        if n_total == 0:
            ca = self.dynamic_coupling_array(fixed_times, leg_x) if dyn else None
            val = self.evaluate(fixed_times, directions, cache, coupling_array=ca)
            return (
                _real_or_raise(val, self._e_psi, where=" (qmc, zero-dimensional)"),
                0.0,
            )

        # bits=64 rather than scipy's default 30: 30-bit points are multiples
        # of 2^-30, so the sample mean carries a left-Riemann bias
        # -(f(1) - f(0)) / 2^31, the same for every seed and every sample
        # count (1.4e-09 relative on demo 7's two-time channel, where the
        # statistical error is smaller).
        u = qmc.Sobol(d=n_total, seed=seed, bits=64).random(n_samples)
        bounds = _causal_bounds(
            spatial, int_vars_pf, ext_integrated, fixed_times, lambda_f, t_ceiling, t_min
        )
        sampled, jacobians = _causal_times(
            bounds, u, {var: i for i, var in enumerate(sobol_vars)}
        )

        values = np.zeros(n_samples)
        for s, (row, jacobian) in enumerate(zip(sampled.tolist(), jacobians.tolist())):
            if jacobian == 0.0:
                continue
            times = {**fixed_times, **dict(zip(sobol_vars, row))}
            ca = self.dynamic_coupling_array(times, leg_x) if dyn else None
            result = self.evaluate(times, directions, cache, coupling_array=ca)
            val = _real_or_raise(result, self._e_psi, where=" (qmc)")
            values[s] = val * jacobian

        return (float(np.mean(values)), _qmc_std_error(values))

    def integrate_moment_qmc_vectorized(
        self,
        lambda_f: float,
        cache: PropagatorCache,
        t_min: float | None = None,
        direction: Any = 0,
        n_samples: int = 2**14,
        seed: int | None = None,
        positions: dict[str, float] | None = None,
        integrate_over: Any = None,
        external_times: dict[str, float] | None = None,
        qmc_coupling_chunk_size: int | None = None,
    ) -> tuple[float, float]:
        """Vectorized QMC integration, with no Python loop over samples.

        Same algorithm as :meth:`integrate_moment_qmc` but evaluates
        all Sobol samples simultaneously using batch propagator lookups.
        Requires ``precompute_C_table`` for the batch C evaluation.

        For the ``iso_R + diag_C`` case with scalar coupling (no
        propagator indices), the integrand at each sample is::

            coupling × prod_R R(t_l, t_r) × prod_C C_diag(t_l, t_r)

        where all R and C values are looked up in batch.

        Args:
            qmc_coupling_chunk_size: Optional positive maximum rows per
                dynamic coupling batch. Every raw symbol callback must be
                sample-separable with stable dtype and trailing shape; a
                ``vectorized=True`` attribute alone does not establish this.
                ``None`` keeps the original single call. Static couplings,
                zero-dimensional integrals and caps at least ``n_samples``
                keep their original path. R/C, contracted coupling output,
                samples and final statistics still use the complete batch.
                See :doc:`/user_guide/workflow` for callback and memory limits.
            positions: Optional mapping ``{point_name: spatial_coord}``
                (e.g. ``{'x': 0.0, 'y': 0.5}``).  When the cache has a
                spatial table built (any :attr:`~PropagatorCache.homogeneity`
                kind), the x-coordinate of each propagator endpoint is
                derived from its direction group's external-point
                position and flows through ``cache.C_at_batch``.  A
                zero-dimensional diagram passes it to ``C_value``.  A
                time-only table has no x-dependence and does not read it.
                A callable coupling reads it on every cache.
            integrate_over: Controls which **external** points have
                their time integrated over ``[t_min, lambda_f]``.

                - ``None`` (default, physics observable):
                  **all externals held fixed at ``lambda_f``**.
                  This gives the equal-time correlator
                  ``⟨φ(t_f) φ(t_f)⟩`` that demo notebooks and MC
                  data compare against.
                - ``"all"``: all externals integrated, giving the
                  time-integrated moment
                  ``⟨∫₀^{t_f}φ(t) dt · ∫₀^{t_f}φ(t') dt'⟩``.
                  Useful e.g. for weak-lensing line-of-sight
                  integrals of the Sachs field.
                - Iterable of external point names (subset):
                  those listed are integrated, the rest fixed at
                  ``lambda_f``.  Enables mixed observables such as
                  an integrated source × a detector field.

                Internal vertex times are always integrated; their
                causal parent-map uses the (possibly fixed)
                external times as upper bounds.

        Returns:
            ``(estimate, std_error)``
        """
        from scipy.stats import qmc

        qmc_coupling_chunk_size = _validate_qmc_coupling_chunk_size(
            qmc_coupling_chunk_size,
        )
        t_min = self._admit(cache, t_min, positions, direction)
        spatial = self.spatial
        int_vars_pf = list(reversed(spatial.time_integration_vars))
        ext_integrated, fixed_times, t_ceiling = self._split_externals(
            integrate_over, lambda_f, external_times
        )

        # Sobol-dimensioned vars = integrated externals + internals.
        sobol_vars = ext_integrated + int_vars_pf
        n_total = len(sobol_vars)

        if n_total == 0:
            # No integration variables, but callable couplings can still be
            # dynamic after R-absorption aliases their legs to fixed externals.
            val = self._evaluate_zero_dimensional(
                lambda_f,
                cache,
                direction=direction,
                positions=positions,
                external_times=external_times,
            )
            return (val, 0.0)

        # Sobol samples (bits=64: see integrate_moment_qmc)
        u = qmc.Sobol(d=n_total, seed=seed, bits=64).random(n_samples)
        columns = {var: i for i, var in enumerate(sobol_vars)}
        times_arr, jacobians = _causal_times(
            _causal_bounds(
                spatial,
                int_vars_pf,
                ext_integrated,
                fixed_times,
                lambda_f,
                t_ceiling,
                t_min,
            ),
            u,
            columns,
        )
        time_of = _batch_time_reader(
            times_arr, columns, fixed_times, lambda_f, spatial.equal_time_aliases
        )

        where = "DiagramIntegrand.integrate_moment_qmc_vectorized"
        dynamic = self.dynamic_coupling
        chunked = None
        if (
            dynamic is not None
            and qmc_coupling_chunk_size is not None
            and qmc_coupling_chunk_size < n_samples
        ):

            def chunked(label_t, label_x):
                return _qmc_couplings_in_chunks(
                    dynamic, label_t, label_x, n_samples, qmc_coupling_chunk_size
                )

        values = self._values_at(
            cache,
            time_of,
            jacobians,
            self._c_lookup(cache, positions, direction, where),
            None
            if dynamic is None
            else self._leg_positions(positions, direction, where=where),
            tag=" (dynamic coupling, qmc)",
            couplings=chunked,
        )
        estimate = _finite_or_raise(float(np.mean(values)), " (qmc_vectorized)")
        return (estimate, _qmc_std_error(values))

    def integrate_moment_gauss_legendre(
        self,
        lambda_f: float,
        cache: PropagatorCache,
        t_min: float | None = None,
        direction: Any = 0,
        n_gauss: int = 8,
        positions: dict[str, float] | None = None,
        integrate_over: Any = None,
        external_times: dict[str, float] | None = None,
        _extra_orderings: tuple = (),
        _extra_bounds: tuple = (),
        gl_chunk_size: int | None = DEFAULT_GL_CHUNK_SIZE,
    ) -> tuple[float, float]:
        """Tensor-product Gauss-Legendre quadrature on the causal
        simplex.

        The integrand is kinked where two times that the causal structure
        leaves unordered cross, if they are the two parents of one variable
        (a vertex with several ψ legs at one time), the two ends of a C
        propagator when C has a derivative jump on its time diagonal (white
        noise, a κ² with a ``|Δt|`` cusp, or a closed-form C that sets
        ``has_diagonal_kink`` or, unset, shows the jump to a numerical
        probe; see :func:`_c_has_diagonal_kink`), or two
        time arguments of a coupling callable that sets
        ``has_coincident_time_kinks`` (see
        :func:`_coupling_kink_candidates`).  The domain is then split at
        those diagonals and the results are added.  Each piece is smooth,
        and the convergence in ``n_gauss`` stays exponential instead of
        ``n^-2``.

        Two ends of a kink, two kinds of piece:

        * both ends integration variables: each consistent order of the
          pairs is its own causal sub-simplex (``_extra_orderings``), up to
          ``k!`` pieces for ``k`` mutually unordered times;
        * one end an external point pinned at a fixed time ``t*``: the
          variable's range ``[t_min, min(parents)]`` is cut in two,
          ``[t*, min(parents)]`` and ``[t_min, min(t*, parents)]``
          (``_extra_bounds``, as ``(variable, "le" | "ge", t*)``), two
          pieces per cut.  With every external at one time the cut lies on
          the boundary of the domain and is skipped, so only unequal
          external times pay for it.

        Both repeat inside each piece until no kink is left.

        Mirrors :meth:`integrate_moment_qmc_vectorized` node-for-node
        with the SAME causal-simplex mapping (parents → upper bounds →
        Jacobians) and the SAME vectorised batch path through
        :meth:`PropagatorCache.R_time_batch` and
        :meth:`PropagatorCache.C_at_batch`, but replaces the Sobol
        ``n_samples`` quasi-random nodes with the deterministic
        ``n_gauss^d`` tensor product of 1-D Gauss-Legendre nodes
        (mapped from ``[-1, 1]`` to ``[0, 1]``).

        For smooth integrands (a finite product of exponentials,
        the typical R/C/κ kernel structure) this gives **exponential
        convergence in n_gauss**, outperforming Sobol QMC at
        modest dimensionality.  In particular, demo2's FK channel
        (4D smooth integrand on a causal simplex of area
        ``t_f^2 / 2``) is dominated at large ``t_f`` by a narrow
        peak band of area ``~ σ_t/γ`` near the upper-right corner;
        Sobol QMC under-resolves it unless ``n_samples`` is very large,
        while ``n_gauss=8`` (4096 nodes for d=4) recovers the
        notebook's hand-derived value to ~5 sig figs.

        **Cost trade-off.**  Tensor-product GL scales as
        ``n_gauss^d``: fine for ``d ≤ 5`` (demo2 FK), fast at
        ``d ≤ 4``, and slow at ``d ≥ 7``.  For high-d integrands
        prefer ``method='qmc_vectorized'`` with a large ``n_samples``
        instead.

        Args:
            n_gauss: Number of GL nodes per dimension.  ``n=8`` is
                a good default (handles up to degree-15 polynomial
                exactly).
            positions, integrate_over: Same as
                :meth:`integrate_moment_qmc_vectorized`.
            gl_chunk_size: Positive maximum nodes per batch (default
                :data:`DEFAULT_GL_CHUNK_SIZE`, 4096), or ``None`` for the
                eager tensor grid at any size.  A grid of at most
                ``gl_chunk_size`` nodes is the eager grid.  Larger grids are
                evaluated in blocks: every node and kink piece is kept and
                the sample-batch workspace is bounded, but callback batch
                sizes and floating-point reduction order change, so
                vectorised hooks must return samplewise results independent
                of batch length.

        Returns:
            ``(estimate, 0.0)``.  GL is deterministic, so there is
            no statistical error to report.  The 0.0 mirrors the
            return shape of the QMC variants for downstream code; it is a
            placeholder, not an estimate or certificate of quadrature error
            or convergence.
        """
        gl_chunk_size = _validate_gl_chunk_size(gl_chunk_size)
        t_min = self._admit(cache, t_min, positions, direction)
        spatial = self.spatial
        int_vars_pf = list(reversed(spatial.time_integration_vars))
        ext_integrated, fixed_times, t_ceiling = self._split_externals(
            integrate_over, lambda_f, external_times
        )
        n_total = len(ext_integrated) + len(int_vars_pf)

        if n_total == 0:
            # No integration variables, but callable couplings can still be
            # dynamic after R-absorption aliases their legs to fixed externals.
            val = self._evaluate_zero_dimensional(
                lambda_f,
                cache,
                direction=direction,
                positions=positions,
                external_times=external_times,
            )
            return (val, 0.0)

        time_orderings = list(spatial.time_orderings) + list(_extra_orderings)
        c_kink = self._c_kink_at(cache, t_min, t_ceiling, positions, direction)
        split = self._sum_kink_pieces(
            lambda orderings, cuts: self.integrate_moment_gauss_legendre(
                lambda_f,
                cache,
                t_min=t_min,
                direction=direction,
                n_gauss=n_gauss,
                positions=positions,
                integrate_over=integrate_over,
                external_times=external_times,
                _extra_orderings=orderings,
                _extra_bounds=cuts,
                gl_chunk_size=gl_chunk_size,
            ),
            c_kink,
            time_orderings,
            fixed_times,
            t_min,
            t_ceiling,
            lambda_f,
            ext_integrated,
            _extra_orderings,
            _extra_bounds,
        )
        if split is not None:
            return split
        if _extra_orderings:
            int_vars_pf = list(
                reversed(
                    _topological_sort_times(
                        tuple(spatial.time_integration_vars), time_orderings
                    )
                )
            )

        u_1d, w_1d = _gauss_legendre_unit_rule(n_gauss)
        if gl_chunk_size is not None and int(n_gauss) ** n_total <= gl_chunk_size:
            gl_chunk_size = None  # one block is the eager grid

        # The causal mapping of the QMC path, with this piece's orderings
        # and cuts.
        bounds = _causal_bounds(
            spatial,
            int_vars_pf,
            ext_integrated,
            fixed_times,
            lambda_f,
            t_ceiling,
            t_min,
            orderings=time_orderings,
            cuts=_applied_cut_bounds(
                spatial, time_orderings, _extra_bounds, swept=ext_integrated
            ),
        )
        columns = {var: i for i, var in enumerate(ext_integrated + int_vars_pf)}
        where = "DiagramIntegrand.integrate_moment_gauss_legendre"
        lookup_C = self._c_lookup(cache, positions, direction, where)
        leg_x = (
            None
            if self.dynamic_coupling is None
            else self._leg_positions(positions, direction, where=where)
        )

        # One accumulator per final kink/causal leaf, matching the eager
        # batch's phase scope; no second evaluation of user callables.
        reality = (
            _BatchRealityAccumulator()
            if gl_chunk_size is not None and self.dynamic_coupling is not None
            else None
        )

        def evaluate_nodes(u, node_weights):
            times_arr, jacobians = _causal_times(bounds, u, columns)
            time_of = _batch_time_reader(
                times_arr, columns, fixed_times, lambda_f, spatial.equal_time_aliases
            )
            values = self._values_at(
                cache,
                time_of,
                jacobians,
                lookup_C,
                leg_x,
                tag=" (dynamic coupling, gauss-legendre)",
                reality=reality,
            )
            # Each tensor-product node carries weight ``w_i = prod_d w_1d[i_d]``,
            # so the integral on the unit cube is a weighted sum, not the QMC
            # mean.  ``values`` already includes the causal-simplex Jacobian.
            return float(np.sum(values * node_weights))

        blocks = _gl_tensor_blocks(u_1d, w_1d, n_total, gl_chunk_size)
        if gl_chunk_size is None:
            u, w = next(blocks)
            estimate = evaluate_nodes(u, w)
        else:
            estimate = _sum_gl_block_totals(evaluate_nodes(u, w) for u, w in blocks)
        if reality is not None:
            reality.validate()
        return (_finite_or_raise(estimate, " (gauss-legendre)"), 0.0)

    def integrate_moment_nquad(
        self,
        lambda_f: float,
        cache: PropagatorCache,
        t_min: float | None = None,
        direction: Any = 0,
        positions: dict[str, float] | None = None,
        integrate_over: Any = None,
        external_times: dict[str, float] | None = None,
        _extra_orderings: tuple = (),
        _extra_bounds: tuple = (),
    ) -> tuple[float, float]:
        """Integrate over time variables using nested adaptive
        quadrature.

        The time domain is split where the integrand is kinked, as in
        :meth:`integrate_moment_gauss_legendre`: each consistent order of
        the kink pairs is integrated as its own causal sub-simplex
        (``_extra_orderings``), a kink against an external point pinned at
        a fixed time cuts the variable's range in two (``_extra_bounds``),
        the split repeats inside each piece until no kink is left, and the
        values and error estimates are added.  Each piece is smooth, so the
        adaptive rule reaches its tolerance without subdividing along a
        kink.  A diagram with no kink takes the unsplit path unchanged.

        See :meth:`integrate_moment_qmc_vectorized` for the
        ``integrate_over`` kwarg semantics (external-time partition
        into integrated vs fixed-at-``lambda_f``).

        A spacetime-dependent (callable) coupling is evaluated at every
        point the quadrature visits, through
        :meth:`dynamic_coupling_array`, as the scalar QMC loop does: the
        callable is called once per integrand evaluation and per coupling
        symbol (one per leg order and leg set, see
        :meth:`~sft_wick.perturbation.DiagramTerm.build_integrand`).
        """
        from scipy.integrate import nquad as _nquad

        t_min = self._admit(cache, t_min, positions, direction)
        spatial = self.spatial
        int_vars = list(spatial.time_integration_vars)
        ext_integrated, fixed_times, t_ceiling = self._split_externals(
            integrate_over, lambda_f, external_times
        )

        directions = self._resolve_group_x(
            spatial,
            positions,
            direction,
            where="DiagramIntegrand.integrate_moment_nquad",
        )

        if not int_vars and not ext_integrated:
            val = self._evaluate_zero_dimensional(
                lambda_f,
                cache,
                direction=direction,
                positions=positions,
                external_times=external_times,
            )
            return (val, 0.0)

        # A spacetime-dependent (callable) coupling leaves the static
        # ``coupling_array`` as a zeros placeholder, so every point the
        # quadrature visits materialises its own coupling tensor, as the
        # scalar QMC loop does.  Up to 0.5.0 this method refused callable
        # couplings instead (before 2026-04 it silently returned 0).
        dyn = self.dynamic_coupling is not None
        leg_x = (
            self._leg_positions(
                positions, direction, where="DiagramIntegrand.integrate_moment_nquad"
            )
            if dyn
            else None
        )

        # Split at kinks until none is left (see
        # integrate_moment_gauss_legendre).  An extra ordering bounds its
        # earlier variable by the later one, so the internals are re-sorted
        # innermost-first under the orderings added so far.
        time_orderings = list(spatial.time_orderings) + list(_extra_orderings)
        c_kink = self._c_kink_at(cache, t_min, t_ceiling, positions, direction)
        split = self._sum_kink_pieces(
            lambda orderings, cuts: self.integrate_moment_nquad(
                lambda_f,
                cache,
                t_min=t_min,
                direction=direction,
                positions=positions,
                integrate_over=integrate_over,
                external_times=external_times,
                _extra_orderings=orderings,
                _extra_bounds=cuts,
            ),
            c_kink,
            time_orderings,
            fixed_times,
            t_min,
            t_ceiling,
            lambda_f,
            ext_integrated,
            _extra_orderings,
            _extra_bounds,
        )
        if split is not None:
            return split
        if _extra_orderings:
            int_vars = _topological_sort_times(
                tuple(spatial.time_integration_vars), time_orderings
            )
        bounds = _causal_bounds(
            spatial,
            int_vars,
            ext_integrated,
            fixed_times,
            lambda_f,
            t_ceiling,
            t_min,
            orderings=time_orderings,
            cuts=_applied_cut_bounds(
                spatial, time_orderings, _extra_bounds, swept=ext_integrated
            ),
        )
        # Quadrature variables = internals + integrated externals.  scipy's
        # nquad integrates index 0 innermost, and a variable can only be
        # bounded by one at a HIGHER index, so a swept-to-swept causal ordering
        # needs the EARLIER external placed last.  The bounds draw them
        # earliest-first; with no such ordering they stay in the caller's
        # order.
        swept = bounds[: len(ext_integrated)]
        ext_ordered = [b.var for b in swept]
        if any(b.lower for b in swept):
            ext_ordered.reverse()
        all_vars = int_vars + ext_ordered

        def f(*args: float) -> float:
            times = dict(fixed_times)
            for i, var in enumerate(all_vars):
                times[var] = args[i]
            ca = self.dynamic_coupling_array(times, leg_x) if dyn else None
            result = self.evaluate(times, directions, cache, coupling_array=ca)
            return _real_or_raise(result, self._e_psi, where=" (nquad)")

        val, err = _nquad(f, _nquad_ranges(bounds, all_vars))
        return (val, err)


def _native_dense_integrand(integrand):
    """Whether the ``"einsum"`` backend may contract this integrand.

    The backend accepts the package's own classes only.  Replacing their
    methods or the module's helpers at run time is unsupported.
    """
    from .perturbation import DiagramTerm

    return (
        type(integrand) is DiagramIntegrand
        and type(integrand.diagram_term) is DiagramTerm
    )


def _native_execution_integrand(integrand):
    if _native_dense_integrand(integrand):
        return True
    module = sys.modules.get("sft_wick._factorized_coupling")
    return module is not None and module.native_factorized_integrand(integrand)


def _build_execution_integrand(
    dt,
    coupling_values,
    fixed_indices,
    contraction_backend="ordered",
    contraction_max_bytes=DEFAULT_CONTRACTION_MAX_BYTES,
):
    from .perturbation import (
        DiagramTerm,
        _NATIVE_BUILD_INTEGRAND,
        _NATIVE_BUILD_INTEGRAND_CODE,
    )

    from types import MethodType

    builder = dt.build_integrand
    if (
        contraction_backend == "factorized"
        and type(dt) is DiagramTerm
        and type(builder) is MethodType
        and builder.__self__ is dt
        and builder.__func__ is _NATIVE_BUILD_INTEGRAND
        and "build_integrand" not in object.__getattribute__(dt, "__dict__")
        and DiagramTerm.__dict__.get("build_integrand") is _NATIVE_BUILD_INTEGRAND
        and _NATIVE_BUILD_INTEGRAND.__code__ is _NATIVE_BUILD_INTEGRAND_CODE
    ):
        return builder(
            coupling_values,
            fixed_indices,
            contraction_backend=contraction_backend,
            contraction_max_bytes=contraction_max_bytes,
        )
    return builder(coupling_values, fixed_indices)


def _execution_options(
    propagator_cache_bytes=None,
    propagator_cache_contract=None,
    contraction_backend="ordered",
    contraction_max_bytes=DEFAULT_CONTRACTION_MAX_BYTES,
):
    """Validate before callbacks; omit default keywords for compatible adapters."""
    validate_cache_options(propagator_cache_bytes, propagator_cache_contract)
    validate_contraction_options(contraction_backend, contraction_max_bytes)
    choices = {}
    if propagator_cache_bytes is not None:
        choices["propagator_cache_bytes"] = propagator_cache_bytes
    if propagator_cache_contract is not None:
        choices["propagator_cache_contract"] = propagator_cache_contract
    if contraction_backend != "ordered":
        choices["contraction_backend"] = contraction_backend
    if contraction_max_bytes != DEFAULT_CONTRACTION_MAX_BYTES:
        choices["contraction_max_bytes"] = contraction_max_bytes
    return choices


@contextmanager
def _propagator_scope(cache, budget=None, contract=None):
    wrapped = wrap_propagator_cache(cache, budget, contract)
    try:
        yield wrapped
    finally:
        if wrapped is not cache:
            wrapped.close()


def _cache_supports_batch_c(cache: "PropagatorCache") -> bool:
    """Whether ``cache`` can evaluate C propagators in batch.

    True when the cache either has a pre-computed spline table
    (``_c_splines is not None`` on :class:`PropagatorCache`) or is a
    custom cache with a native batch implementation that sets
    ``_c_splines`` to a truthy sentinel (e.g. the analytical cache
    used in ``examples/demo1`` and ``tests/test_deductive_numerics``).
    """
    return getattr(cache, "_c_splines", None) is not None


def _cache_has_spatial_table(cache: "PropagatorCache") -> bool:
    """Whether ``cache`` has been equipped with any spatial
    (x-aware) table: full or lazy, any homogeneity kind.

    Used by the integrators to decide whether to route each C
    propagator through the spatial-aware ``C_at_batch`` path or the
    legacy ``C_diagonal_batch`` path (which ignores x).  False when
    the cache was built with only the legacy ``precompute_C_table``
    or when it is a custom cache that doesn't implement the new
    spatial attributes.  In both cases the legacy path is
    bit-identical to pre-Phase-5 behaviour.
    """
    if getattr(cache, "_closed_form_only", False):
        # Closed-form-only mode: no spline at all, but the
        # ``C_at_batch`` override IS spatial-aware (it routes the
        # per-sample positions straight into the user's c_fn).
        return True
    return _built_spatial_table(cache)


def _time_only_table(cache: "PropagatorCache") -> bool:
    """Whether ``cache`` reads C from a table that has no position in it."""
    return getattr(cache, "_c_splines", None) is not None and not (
        _cache_has_spatial_table(cache)
    )


def _require_time_table_holds(cache: "PropagatorCache", pairs: list[tuple]) -> None:
    """Raise ``ValueError`` when a time-only C table would stand for a C it
    does not hold.

    Such a table holds C at one position, ``direction`` of
    :meth:`PropagatorCache.precompute_C_table`, at both ends.  It stands for
    the C at another pair of positions when κ² and σ² read the same there
    (:func:`_noise_reads_alike`).  A kernel that cannot be read at either pair
    is not judged.  A pair that passes is remembered on the cache, because
    :meth:`DiagramIntegrand.evaluate` asks at every call.
    """
    at_table = getattr(cache, "_c_table_direction", 0.0)
    seen = getattr(cache, "_time_table_reads", None)
    if seen is None or len(seen) >= _C_KINK_MEMO_SIZE:
        seen = cache._time_table_reads = set()
    for x_l, x_r in pairs:
        try:
            key = tuple(
                np.asarray(x, dtype=float).tobytes() for x in (at_table, x_l, x_r)
            ) + (cache._c_table_range,)
            if key in seen:
                continue
            alike = _noise_reads_alike(
                cache.model, (x_l, x_r), (at_table, at_table), cache._c_table_range
            )
        except Exception:  # noqa: BLE001 (a user callable or a stand-in)
            continue
        if not alike:
            raise ValueError(
                f"this cache has a time-only C table, built at position "
                f"{at_table!r}, and the noise reads differently at positions "
                f"{x_l!r} and {x_r!r} on a C propagator.  Build a spatial "
                f"table (precompute_C_table_translation / _rotation / "
                f"_general)."
            )
        seen.add(key)


def _built_spatial_table(cache: "PropagatorCache") -> bool:
    """Whether ``cache`` holds a spatial table, full or lazy, of any
    homogeneity kind.  A closed-form-only cache holds none."""
    names = (
        "_c_translation_splines",
        "_lazy_translation",
        "_c_rotation_splines",
        "_lazy_rotation",
        "_c_general_interpolators",
        "_lazy_general",
    )
    return any(getattr(cache, n, None) is not None for n in names)


@lru_cache(maxsize=1)
def _package_batch_methods() -> dict:
    """The batch accessors of the package's own caches, each mapped to the
    methods whose output it returns.

    Each allocates the array it returns, except
    :meth:`PropagatorCache.C_at_batch`, which returns the array of
    ``_assemble_C``, ``_lazy_lookup`` or ``_closed_form_at_batch_diag``: it
    counts as the package's own only while those are (:func:`_trusted`).
    Imported on first use, since :mod:`sft_wick.spectral` imports this
    module.
    """
    from .spectral import SpectralPropagatorCache

    return {
        PropagatorCache.C_diagonal_batch: (),
        PropagatorCache.C_matrix_batch: (),
        PropagatorCache.C_at_batch: (
            PropagatorCache._assemble_C,
            PropagatorCache._lazy_lookup,
            PropagatorCache._closed_form_at_batch_diag,
        ),
        PropagatorCache.R_matrix_batch: (),
        SpectralPropagatorCache.C_diagonal_batch: (),
    }


def _trusted(method: Any, registry: dict) -> bool:
    """Whether the bound ``method`` is one of the package's own accessors in
    ``registry`` and every method it delegates to is the package's own on
    the same object.

    A subclass can keep a package accessor and override what it calls:
    ``SpectralPropagatorCache.C_value`` passes its positions to
    ``_c_matrix``, and ``PropagatorCache.C_at_batch`` returns the array of
    ``_assemble_C``, ``_lazy_lookup`` or ``_closed_form_at_batch_diag``.
    Such an accessor is not trusted, so its arguments or its output are
    copied as for any user-written method.
    """
    delegates = registry.get(getattr(method, "__func__", None))
    if delegates is None:
        return False
    return not _overrides(method.__self__, *delegates)


def _overrides(obj: Any, *funcs: Any) -> bool:
    """Whether ``obj`` resolves the name of any of ``funcs``, functions of
    the package's own classes, to something else: a subclass's method or
    an instance attribute."""
    for func in funcs:
        attr = getattr(obj, func.__name__, None)
        if getattr(attr, "__func__", attr) is not func:
            return True
    return False


def _owned_lookup(cache: Any, func: Any, out: Any) -> Any:
    """``out``, the array a package accessor got from ``cache.<func>``, or
    a copy of it when ``cache`` overrides ``func``.

    ``C_value`` and ``C_diagonal`` return fresh arrays or read-only memo
    entries, so a caller may keep them.  A subclass's legacy-table lookup
    (``_C_value_from_table``, ``_C_diagonal_from_table``) may fill and
    return one buffer, which would change every result kept from an
    earlier call.  The package's own lookups allocate and are not copied.
    """
    if _overrides(cache, func):
        return np.array(out, copy=True)
    return out


def _owned_batch(cache: Any, name: str, out: Any) -> Any:
    """``out``, returned by ``cache.<name>``, or a copy of it when that
    method is not one of the package's own.

    The batched integrators keep one batch per C factor and per matrix R
    factor while they ask the cache for the next one.  A user-written cache
    (a subclass overriding the method or a method it returns the output of,
    or a cache that is not a :class:`PropagatorCache`) may fill and return
    one buffer, which would turn every kept factor into the last one.  The
    package's own methods return fresh arrays and are not copied
    (:func:`_trusted`).  The propagator lookup bank is looked through: it
    returns its source's array on a miss and a copy on a hit.
    """
    source = cache._source if type(cache) is _PropagatorLookupCache else cache
    if _trusted(getattr(source, name, None), _package_batch_methods()):
        return out
    return np.array(out, copy=True)


@lru_cache(maxsize=1)
def _package_c_lookups() -> dict:
    """The scalar C accessors of the package's own caches, each mapped to
    the methods it passes its positions to.

    Each reads its position arguments only, or passes copies on to a method
    that may write into them (:meth:`PropagatorCache.C_diagonal` calling an
    overridden ``C_value``, through :func:`_c_lookup_positions`), except
    :meth:`SpectralPropagatorCache.C_value`, which passes them to
    ``_c_matrix``: it counts as the package's own only while ``_c_matrix``
    is (:func:`_trusted`).  Imported on first use, as in
    :func:`_package_batch_methods`.
    """
    from .spectral import SpectralPropagatorCache

    return {
        PropagatorCache.C_value: (),
        PropagatorCache.C_diagonal: (),
        SpectralPropagatorCache.C_value: (SpectralPropagatorCache._c_matrix,),
        SpectralPropagatorCache.C_diagonal: (),
    }


def _c_lookup_positions(cache: Any, name: str, n1: Any, n2: Any) -> tuple[Any, Any]:
    """The two positions to pass to ``cache.<name>``, which is ``C_value``
    or ``C_diagonal``.

    The scalar evaluation passes the caller's position objects, and both
    ends of a C inside one direction group are one object.  A user-written
    method that formed the separation in place (``n1 -= n2``) would change
    the caller's positions and every later lookup.  Such a method, or a
    package method that hands its positions to an overridden one (a
    ``SpectralPropagatorCache`` subclass overriding ``_c_matrix``), receives
    copies of array and list positions (:func:`_own_copy`).  The package's
    own methods receive the objects themselves.  The propagator lookup bank
    forwards these two methods unchanged, so the bound method found here is
    the one that runs.
    """
    if _trusted(getattr(cache, name, None), _package_c_lookups()):
        return n1, n2
    return _own_copy(n1), _own_copy(n2)


def _r_product_times(cache: Any, times: dict) -> dict:
    """The times dict to pass to ``cache.R_product``.

    :meth:`DiagramIntegrand.evaluate` reads ``times`` again for every C
    lookup after the R product, and a direct caller may read its dict
    again after the call.  A user-written ``R_product`` that wrote into the
    dict (``times[k] -= t_min``, ``pop``, ``update``) would change those
    lookups with no error, so it receives a copy.  The package's own reads
    the dict only and receives it itself; the propagator lookup bank
    forwards ``R_product`` unchanged.
    """
    method = getattr(cache, "R_product", None)
    if getattr(method, "__func__", None) is PropagatorCache.R_product:
        return times
    return dict(times)


def _time_table_C_batch(cache: Any, t_l: np.ndarray, t_r: np.ndarray) -> np.ndarray:
    """C from a cache without a spatial table: ``(n, N)`` diagonals from
    ``C_diagonal_batch``, or ``(n, N, N)`` from ``C_matrix_batch`` when the
    model keeps the off-diagonal entries (``diag_C=False``).

    The integrators keep the result while they look up the next C, so the
    output of a user-written batch method is copied (:func:`_owned_batch`).

    Raises:
        ValueError: if the model keeps off-diagonal entries but the cache
            has no ``C_matrix_batch``; ``C_diagonal_batch`` would drop them.
    """
    model = getattr(cache, "model", None)
    if getattr(model, "diag_C", True):
        return _owned_batch(cache, "C_diagonal_batch", cache.C_diagonal_batch(t_l, t_r))
    full = getattr(cache, "C_matrix_batch", None)
    if full is None:
        raise ValueError(
            f"{type(cache).__name__}: the propagator model keeps off-diagonal "
            f"C entries (diag_C=False), but the cache has no C_matrix_batch, "
            f"and C_diagonal_batch returns C_aa only, so C_ab with a != b "
            f"would be dropped.  Give the cache a C_matrix_batch(t1, t2) -> "
            f"(n, N, N), or a spatial table."
        )
    return _owned_batch(cache, "C_matrix_batch", full(t_l, t_r))


def _C_at_fixed_positions(
    cache: Any, t_l: np.ndarray, t_r: np.ndarray, x_l: Any, x_r: Any
) -> np.ndarray:
    """``cache.C_at_batch`` with one fixed position per side.

    ``C_at_batch`` reads a 1-D array whose length equals the batch as one
    scalar position per sample, so a fixed d-vector on a batch of d > 1
    samples would be misread that way.  Such a vector is tiled to
    ``(n, d)`` first, as ``C_at_batch`` tiles a vector of another length.
    ``C_at_batch`` already reads a vector of another length as fixed, and
    with ``n = d = 1`` both readings give the same value.

    Any other array or list position is passed as an owned float copy.  The
    caller's positions are read again by later lookups and diagrams, and
    both ends of a C inside one group are one object, so a cache that wrote
    into its position arguments (``x1 -= x2``) would change them.  Scalars
    pass unchanged.

    The integrators keep the result while they look up the next C, so the
    output of a user-written ``C_at_batch`` is copied (:func:`_owned_batch`).
    """
    n = len(np.atleast_1d(t_l))

    def rows(x: Any) -> Any:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1 and n > 1 and len(arr) == n:
            return np.tile(arr, (n, 1))
        if isinstance(x, (np.ndarray, list)):
            return np.array(arr, copy=True)
        return x

    return _owned_batch(
        cache, "C_at_batch", cache.C_at_batch(t_l, t_r, rows(x_l), rows(x_r))
    )


def integrate_moment(
    integrand: DiagramIntegrand,
    lambda_f: float,
    cache: PropagatorCache,
    t_min: float | None = None,
    direction: Any = 0,
    method: str = "qmc",
    n_samples: int = 2**14,
    seed: int | None = None,
    positions: dict[str, float] | None = None,
    integrate_over: Any = None,
    n_gauss: int = 8,
    external_times: dict[str, float] | None = None,
    gl_chunk_size: int | None = DEFAULT_GL_CHUNK_SIZE,
    qmc_coupling_chunk_size: int | None = None,
    propagator_cache_bytes: int | None = None,
    propagator_cache_contract: str | None = None,
    contraction_backend: str = "ordered",
    contraction_max_bytes: int = DEFAULT_CONTRACTION_MAX_BYTES,
) -> tuple[float, float]:
    """Integrate a diagram's contribution over all time variables.

    Convenience wrapper around
    :meth:`DiagramIntegrand.integrate_moment_qmc`,
    :meth:`DiagramIntegrand.integrate_moment_qmc_vectorized`,
    :meth:`DiagramIntegrand.integrate_moment_gauss_legendre`, and
    :meth:`DiagramIntegrand.integrate_moment_nquad`.

    **Dispatch logic.**  With ``method='qmc'`` (default) the function
    auto-selects the fastest QMC path compatible with the supplied
    cache:

    - If the cache supports batched C evaluation (either
      ``PropagatorCache.precompute_C_table`` has been called, or a
      custom cache implements ``C_diagonal_batch`` natively) →
      :meth:`integrate_moment_qmc_vectorized` (~200× faster than
      the scalar loop on typical workloads), for a scalar or a
      matrix-valued R.  Until the batched integrators learned matrix R,
      a matrix R always took the scalar loop here.
    - Otherwise → :meth:`integrate_moment_qmc` (scalar Python loop).

    Users who explicitly want the scalar loop (e.g. for debugging
    or with a non-batch-capable custom cache) can pass
    ``method='qmc_scalar'``.

    For best performance, call
    :meth:`PropagatorCache.precompute_C_table` before integrating so
    the vectorised path is selected automatically.

    .. note::

       "Batch-capable" is a deliberate handshake on ``_c_splines``, so
       a cache carrying only a **spatial** table (rotation /
       translation / general, built by
       ``precompute_C_table_rotation`` and friends) reports False and
       ``method='qmc'`` routes it to the scalar loop, even though
       :meth:`integrate_moment_qmc_vectorized` accepts such a cache
       perfectly well via ``C_at_batch``.  The two paths agree, so this
       costs speed only, but it costs a lot of it (~125× on the
       two-point order-2 rotation workload in
       ``tests/test_diag_fast_component_labels.py``).  Pass
       ``method='qmc_vectorized'`` explicitly for spatial caches; the
       L1 :meth:`Expansion.evaluate` API already defaults to it.

    Args:
        integrand: A :class:`DiagramIntegrand` built from a
            :class:`~sft_wick.perturbation.DiagramTerm`.
        lambda_f: Upper integration limit for external times.
        cache: A :class:`PropagatorCache` (or compatible custom
            cache).  The batched integrators copy the arrays a custom
            cache's batch methods return, so those may reuse one buffer.
            Where ``qmc_vectorized``, ``gauss_legendre`` and
            :func:`integrate_two_point_qmc` read a time batch again for a
            later factor, they pass it read-only, so a batch method that
            writes into it raises ``ValueError``.  ``C_at_batch``, and a
            custom cache's ``C_value`` and ``C_diagonal``, receive copies of
            array positions.
        t_min: Lower time bound; the default is the model's ``t_min``.
        direction: Direction value for all spatial points.
        positions: ``{point_name: position}``, as in
            :meth:`DiagramIntegrand.integrate_moment_qmc_vectorized`.  A
            cache with a time-only table refuses positions that its κ² tells
            apart on a C propagator.
        method: ``'qmc'`` (default, auto-selects vectorised vs
            scalar), ``'qmc_scalar'`` (force scalar loop),
            ``'qmc_vectorized'`` (force vectorised; raises if cache
            doesn't support batch), ``'gauss_legendre'``
            (tensor-product Gauss-Legendre, deterministic), or
            ``'nquad'`` (nested adaptive quadrature).
        n_samples: Number of Sobol samples (QMC only, should be
            a power of 2).
        seed: Random seed for reproducibility (QMC only).
        n_gauss: Gauss-Legendre nodes per dimension
            (``'gauss_legendre'`` only).
        gl_chunk_size: Positive maximum GL nodes per batch (default 4096;
            a grid no larger than one batch is the eager grid); ``None``
            keeps eager GL at any size. Validated but unused by other
            integration methods.
        qmc_coupling_chunk_size: Positive maximum dynamic coupling rows per
            vectorized QMC call; requires sample-separable raw callbacks with
            stable dtype and trailing shape. ``None`` preserves the full batch.
            Validated but unused on scalar QMC, GL and nquad paths.

    Returns:
        ``(estimate, error)`` tuple.


    Execution choices: ``propagator_cache_bytes`` (None/0 disables) requires
    ``propagator_cache_contract="fixed_model"`` when positive. The caller
    promises deterministic fixed propagators for this complete query. Retained
    raw key/output bytes are bounded; process RSS and native tables are not.
    ``contraction_backend="factorized"`` additionally requests original native
    static factor preparation when building diagrams. Public coefficient-array
    access and scalar/unsupported integration can materialize it; known native
    restoration above ``contraction_max_bytes`` raises before allocation.
    Already prepared dense integrands retain their coefficient. See the
    resources guide for exact array semantics and preparation boundaries.

    ``contraction_backend="einsum"`` contracts the prepared static coefficient
    with the propagator batches in pairwise steps. Its result lies within the
    error bound stated in the resources guide, which the ordered fold also
    satisfies, and it falls back to the ordered fold when it cannot keep that
    bound, would raise differently, or is estimated to be slower.
    ``contraction_max_bytes`` bounds planned array workspace, not total memory.
    """
    _execution_options(
        propagator_cache_bytes,
        propagator_cache_contract,
        contraction_backend,
        contraction_max_bytes,
    )
    # Preserve the caller's object lifetime while using a native configured view.
    _original_integrand = integrand
    if _native_execution_integrand(integrand) and (
        integrand._contraction_backend != contraction_backend
        or integrand._contraction_max_bytes != contraction_max_bytes
    ):
        # Custom protocols keep the original object, including its identity.
        # Retain native cached metadata. dataclasses.replace
        # drops non-field attributes, including compatible custom methods.
        configured = object.__new__(type(integrand))
        configured.__dict__.update(integrand.__dict__)
        module = sys.modules.get("sft_wick._factorized_coupling")
        if module is not None:
            module.share_factorized_owner(configured, integrand)
        object.__setattr__(configured, "_contraction_backend", contraction_backend)
        object.__setattr__(configured, "_contraction_max_bytes", contraction_max_bytes)
        integrand = configured
    with _propagator_scope(
        cache, propagator_cache_bytes, propagator_cache_contract
    ) as scoped_cache:
        cache = scoped_cache
        return _integrate_moment_dispatch(
            integrand=integrand,
            lambda_f=lambda_f,
            cache=cache,
            t_min=t_min,
            direction=direction,
            method=method,
            n_samples=n_samples,
            seed=seed,
            positions=positions,
            integrate_over=integrate_over,
            n_gauss=n_gauss,
            external_times=external_times,
            gl_chunk_size=gl_chunk_size,
            qmc_coupling_chunk_size=qmc_coupling_chunk_size,
        )


#: The ``method`` values :func:`_integrate_moment_dispatch` accepts; the
#: YAML parser and the workflow schema read them from here.
INTEGRATION_METHODS = ("qmc_vectorized", "qmc", "qmc_scalar", "gauss_legendre", "nquad")


def _integrate_moment_dispatch(
    integrand: DiagramIntegrand,
    lambda_f: float,
    cache: PropagatorCache,
    t_min: float | None = None,
    direction: Any = 0,
    method: str = "qmc",
    n_samples: int = 2**14,
    seed: int | None = None,
    positions: dict[str, float] | None = None,
    integrate_over: Any = None,
    n_gauss: int = 8,
    external_times: dict[str, float] | None = None,
    gl_chunk_size: int | None = DEFAULT_GL_CHUNK_SIZE,
    qmc_coupling_chunk_size: int | None = None,
) -> tuple[float, float]:
    """Integrate a diagram's contribution over all time variables.

    Convenience wrapper around
    :meth:`DiagramIntegrand.integrate_moment_qmc`,
    :meth:`DiagramIntegrand.integrate_moment_qmc_vectorized`,
    :meth:`DiagramIntegrand.integrate_moment_gauss_legendre`, and
    :meth:`DiagramIntegrand.integrate_moment_nquad`.

    **Dispatch logic.**  With ``method='qmc'`` (default) the function
    auto-selects the fastest QMC path compatible with the supplied
    cache:

    - If the cache supports batched C evaluation (either
      ``PropagatorCache.precompute_C_table`` has been called, or a
      custom cache implements ``C_diagonal_batch`` natively) →
      :meth:`integrate_moment_qmc_vectorized` (~200× faster than
      the scalar loop on typical workloads), for a scalar or a
      matrix-valued R.  Until the batched integrators learned matrix R,
      a matrix R always took the scalar loop here.
    - Otherwise → :meth:`integrate_moment_qmc` (scalar Python loop).

    Users who explicitly want the scalar loop (e.g. for debugging
    or with a non-batch-capable custom cache) can pass
    ``method='qmc_scalar'``.

    For best performance, call
    :meth:`PropagatorCache.precompute_C_table` before integrating so
    the vectorised path is selected automatically.

    .. note::

       "Batch-capable" is a deliberate handshake on ``_c_splines``, so
       a cache carrying only a **spatial** table (rotation /
       translation / general, built by
       ``precompute_C_table_rotation`` and friends) reports False and
       ``method='qmc'`` routes it to the scalar loop, even though
       :meth:`integrate_moment_qmc_vectorized` accepts such a cache
       perfectly well via ``C_at_batch``.  The two paths agree, so this
       costs speed only, but it costs a lot of it (~125× on the
       two-point order-2 rotation workload in
       ``tests/test_diag_fast_component_labels.py``).  Pass
       ``method='qmc_vectorized'`` explicitly for spatial caches; the
       L1 :meth:`Expansion.evaluate` API already defaults to it.

    Args:
        integrand: A :class:`DiagramIntegrand` built from a
            :class:`~sft_wick.perturbation.DiagramTerm`.
        lambda_f: Upper integration limit for external times.
        cache: A :class:`PropagatorCache` (or compatible custom
            cache).  The batched integrators copy the arrays a custom
            cache's batch methods return, so those may reuse one buffer.
            Where ``qmc_vectorized``, ``gauss_legendre`` and
            :func:`integrate_two_point_qmc` read a time batch again for a
            later factor, they pass it read-only, so a batch method that
            writes into it raises ``ValueError``.  ``C_at_batch``, and a
            custom cache's ``C_value`` and ``C_diagonal``, receive copies of
            array positions.
        t_min: Lower time bound; the default is the model's ``t_min``.
        direction: Direction value for all spatial points.
        positions: ``{point_name: position}``, as in
            :meth:`DiagramIntegrand.integrate_moment_qmc_vectorized`.  A
            cache with a time-only table refuses positions that its κ² tells
            apart on a C propagator.
        method: ``'qmc'`` (default, auto-selects vectorised vs
            scalar), ``'qmc_scalar'`` (force scalar loop),
            ``'qmc_vectorized'`` (force vectorised; raises if cache
            doesn't support batch), ``'gauss_legendre'``
            (tensor-product Gauss-Legendre, deterministic), or
            ``'nquad'`` (nested adaptive quadrature).
        n_samples: Number of Sobol samples (QMC only, should be
            a power of 2).
        seed: Random seed for reproducibility (QMC only).
        n_gauss: Gauss-Legendre nodes per dimension
            (``'gauss_legendre'`` only).
        gl_chunk_size: Positive maximum GL nodes per batch (default 4096;
            a grid no larger than one batch is the eager grid); ``None``
            keeps eager GL at any size. Validated but unused by other
            integration methods.
        qmc_coupling_chunk_size: Positive maximum dynamic coupling rows per
            vectorized QMC call; requires sample-separable raw callbacks with
            stable dtype and trailing shape. ``None`` preserves the full batch.
            Validated but unused on scalar QMC, GL and nquad paths.

    Returns:
        ``(estimate, error)`` tuple.
    """
    gl_chunk_size = _validate_gl_chunk_size(gl_chunk_size)
    qmc_coupling_chunk_size = _validate_qmc_coupling_chunk_size(qmc_coupling_chunk_size)
    qmc_options = (
        {}
        if qmc_coupling_chunk_size is None
        else {"qmc_coupling_chunk_size": qmc_coupling_chunk_size}
    )
    if method == "qmc":
        if _cache_supports_batch_c(cache):
            return integrand.integrate_moment_qmc_vectorized(
                lambda_f,
                cache,
                t_min=t_min,
                direction=direction,
                n_samples=n_samples,
                seed=seed,
                positions=positions,
                integrate_over=integrate_over,
                external_times=external_times,
                **qmc_options,
            )
        return integrand.integrate_moment_qmc(
            lambda_f,
            cache,
            t_min=t_min,
            direction=direction,
            n_samples=n_samples,
            seed=seed,
            positions=positions,
            integrate_over=integrate_over,
            external_times=external_times,
        )
    elif method == "qmc_vectorized":
        return integrand.integrate_moment_qmc_vectorized(
            lambda_f,
            cache,
            t_min=t_min,
            direction=direction,
            n_samples=n_samples,
            seed=seed,
            positions=positions,
            integrate_over=integrate_over,
            external_times=external_times,
            **qmc_options,
        )
    elif method == "qmc_scalar":
        return integrand.integrate_moment_qmc(
            lambda_f,
            cache,
            t_min=t_min,
            direction=direction,
            n_samples=n_samples,
            seed=seed,
            positions=positions,
            integrate_over=integrate_over,
            external_times=external_times,
        )
    elif method == "nquad":
        return integrand.integrate_moment_nquad(
            lambda_f,
            cache,
            t_min=t_min,
            direction=direction,
            positions=positions,
            integrate_over=integrate_over,
            external_times=external_times,
        )
    elif method == "gauss_legendre":
        return integrand.integrate_moment_gauss_legendre(
            lambda_f,
            cache,
            t_min=t_min,
            direction=direction,
            n_gauss=n_gauss,
            positions=positions,
            integrate_over=integrate_over,
            external_times=external_times,
            gl_chunk_size=gl_chunk_size,
        )
    else:
        raise ValueError(
            f"Unknown method {method!r}; use one of {list(INTEGRATION_METHODS)}"
        )


def _eval_single_diagram(
    dt,
    coupling_values,
    fixed_indices,
    lambda_f,
    cache,
    t_min,
    direction,
    method,
    n_samples,
    seed,
    positions,
    integrate_over,
    n_gauss,
    external_times=None,
    _prepared_integrand=None,
    gl_chunk_size=DEFAULT_GL_CHUNK_SIZE,
    qmc_coupling_chunk_size=None,
    contraction_backend="ordered",
    contraction_max_bytes=DEFAULT_CONTRACTION_MAX_BYTES,
):
    """Evaluate one diagram term end-to-end (build integrand + integrate).

    Top-level function so it is picklable for multiprocessing.
    """
    ig = (
        _prepared_integrand
        if _prepared_integrand is not None
        else _build_execution_integrand(
            dt,
            coupling_values,
            fixed_indices,
            contraction_backend,
            contraction_max_bytes,
        )
    )
    return integrate_moment(
        ig,
        lambda_f,
        cache,
        **_execution_options(
            contraction_backend=contraction_backend,
            contraction_max_bytes=contraction_max_bytes,
        ),
        t_min=t_min,
        direction=direction,
        method=method,
        n_samples=n_samples,
        seed=seed,
        positions=positions,
        integrate_over=integrate_over,
        n_gauss=n_gauss,
        external_times=external_times,
        gl_chunk_size=gl_chunk_size,
        **(
            {}
            if qmc_coupling_chunk_size is None
            else {"qmc_coupling_chunk_size": qmc_coupling_chunk_size}
        ),
    )


def integrate_diagrams(
    diagram_terms: list,
    coupling_values: dict,
    lambda_f: float,
    cache: "PropagatorCache",
    t_min: float | None = None,
    direction: Any = 0,
    method: str = "qmc",
    n_samples: int = 2**14,
    seed: int | None = None,
    fixed_indices: dict[str, int] | None = None,
    n_jobs: int = 1,
    positions: dict[str, Any] | None = None,
    integrate_over: Any = None,
    n_gauss: int = 8,
    external_times: dict[str, float] | None = None,
    progress_tick: Callable | None = None,
    _prepared_integrands: tuple | None = None,
    gl_chunk_size: int | None = DEFAULT_GL_CHUNK_SIZE,
    qmc_coupling_chunk_size: int | None = None,
    propagator_cache_bytes: int | None = None,
    propagator_cache_contract: str | None = None,
    contraction_backend: str = "ordered",
    contraction_max_bytes: int = DEFAULT_CONTRACTION_MAX_BYTES,
) -> tuple[float, list[tuple[float, float]]]:
    """Integrate a batch of diagram terms, optionally in parallel.

    Builds integrands and evaluates all diagrams, returning the total
    and per-diagram results.  When ``n_jobs != 1``, diagrams are
    evaluated in parallel using :mod:`joblib`.

    .. note::
        ``n_jobs`` defaults to ``1`` (sequential), matching the L1
        ``Expansion.evaluate`` / ``Expansion.sweep`` defaults. Pass
        ``-1`` to use all CPU cores. Sequential and parallel paths
        are bit-identical when the QMC seed is fixed.

    Args:
        diagram_terms: List of :class:`~sft_wick.perturbation.DiagramTerm`.
        coupling_values: Coupling tensor arrays (e.g. ``{'F': F_code}``).
        lambda_f: Upper integration limit.
        cache: A :class:`PropagatorCache`.
        t_min: Lower time bound; the default is the model's ``t_min``.
        direction: Direction value for all spatial points.
        method: Integrator selector accepted by :func:`integrate_moment`.
        gl_chunk_size: Positive GL batch bound (default 4096) or ``None``,
            forwarded to :func:`integrate_moment` in serial and joblib workers.
        qmc_coupling_chunk_size: Optional positive dynamic coupling batch bound
            for vectorized QMC; see :func:`integrate_moment` for the callback
            contract. Each worker constructs its own arrays.
        n_samples: Number of Sobol samples (QMC only).
        seed: Random seed (QMC only).
        fixed_indices: Optional pinned index values for
            :meth:`~sft_wick.perturbation.DiagramTerm.build_integrand`.
        n_jobs: Number of parallel jobs.  ``1`` = sequential
            (default, safe under nested-callers), ``-1`` = all CPU
            cores.  A built-in guard falls
            back to sequential for ``len(diagram_terms) <= 2`` to
            avoid joblib's ~1 s startup overhead on trivial batches.
            Requires :mod:`joblib` when ``n_jobs != 1``.
        progress_tick: Optional ``tick(n=1)`` callable invoked after each
            diagram finishes (per batch on the parallel path).  Used by
            :meth:`~sft_wick.workflow.Expansion.sweep` to drive one bar
            across grid points × diagrams; never affects results.

    Returns:
        ``(total, details)`` where *total* is the scalar sum and
        *details* is a list of ``(estimate, error)`` per diagram.

    Raises:
        ValueError: if a term's R-propagator component indices contradict
            ``cache``'s R type: two different indices against a scalar-R
            cache (``compute_moment``'s default flags), an index-free R
            (``iso_R=True``) against a matrix-R cache, or two different
            indices on an R absorbed into an ``already_R_contracted``
            vertex.  The message names the flags that match the cache.
            Also when ``positions`` places two points of one R-connected
            direction group at different coordinates (see
            ``_require_one_position_per_group``).


    Execution choices: ``propagator_cache_bytes`` (None/0 disables) requires
    ``propagator_cache_contract="fixed_model"`` when positive. The caller
    promises deterministic fixed propagators for this complete query. Retained
    raw key/output bytes are bounded; process RSS and native tables are not.
    ``contraction_backend="factorized"`` additionally requests original native
    static factor preparation when building diagrams. Public coefficient-array
    access and scalar/unsupported integration can materialize it; known native
    restoration above ``contraction_max_bytes`` raises before allocation.
    Already prepared dense integrands retain their coefficient. See the
    resources guide for exact array semantics and preparation boundaries.

    ``contraction_backend="einsum"`` contracts the prepared static coefficient
    with the propagator batches in pairwise steps. Its result lies within the
    error bound stated in the resources guide, which the ordered fold also
    satisfies, and it falls back to the ordered fold when it cannot keep that
    bound, would raise differently, or is estimated to be slower.
    ``contraction_max_bytes`` bounds planned array workspace, not total memory.
    """
    _execution_options(
        propagator_cache_bytes,
        propagator_cache_contract,
        contraction_backend,
        contraction_max_bytes,
    )
    with _propagator_scope(
        cache, propagator_cache_bytes, propagator_cache_contract
    ) as scoped_cache:
        cache = scoped_cache
        return _integrate_diagrams_dispatch(
            diagram_terms=diagram_terms,
            coupling_values=coupling_values,
            lambda_f=lambda_f,
            cache=cache,
            t_min=t_min,
            direction=direction,
            method=method,
            n_samples=n_samples,
            seed=seed,
            fixed_indices=fixed_indices,
            n_jobs=n_jobs,
            positions=positions,
            integrate_over=integrate_over,
            n_gauss=n_gauss,
            external_times=external_times,
            progress_tick=progress_tick,
            _prepared_integrands=_prepared_integrands,
            gl_chunk_size=gl_chunk_size,
            qmc_coupling_chunk_size=qmc_coupling_chunk_size,
            contraction_backend=contraction_backend,
            contraction_max_bytes=contraction_max_bytes,
        )


def _integrate_diagrams_dispatch(
    diagram_terms: list,
    coupling_values: dict,
    lambda_f: float,
    cache: "PropagatorCache",
    t_min: float | None = None,
    direction: Any = 0,
    method: str = "qmc",
    n_samples: int = 2**14,
    seed: int | None = None,
    fixed_indices: dict[str, int] | None = None,
    n_jobs: int = 1,
    positions: dict[str, Any] | None = None,
    integrate_over: Any = None,
    n_gauss: int = 8,
    external_times: dict[str, float] | None = None,
    progress_tick: Callable | None = None,
    _prepared_integrands: tuple | None = None,
    gl_chunk_size: int | None = DEFAULT_GL_CHUNK_SIZE,
    qmc_coupling_chunk_size: int | None = None,
    contraction_backend="ordered",
    contraction_max_bytes=DEFAULT_CONTRACTION_MAX_BYTES,
) -> tuple[float, list[tuple[float, float]]]:
    """Integrate a batch of diagram terms, optionally in parallel.

    Builds integrands and evaluates all diagrams, returning the total
    and per-diagram results.  When ``n_jobs != 1``, diagrams are
    evaluated in parallel using :mod:`joblib`.

    .. note::
        ``n_jobs`` defaults to ``1`` (sequential), matching the L1
        ``Expansion.evaluate`` / ``Expansion.sweep`` defaults. Pass
        ``-1`` to use all CPU cores. Sequential and parallel paths
        are bit-identical when the QMC seed is fixed.

    Args:
        diagram_terms: List of :class:`~sft_wick.perturbation.DiagramTerm`.
        coupling_values: Coupling tensor arrays (e.g. ``{'F': F_code}``).
        lambda_f: Upper integration limit.
        cache: A :class:`PropagatorCache`.
        t_min: Lower time bound; the default is the model's ``t_min``.
        direction: Direction value for all spatial points.
        method: Integrator selector accepted by :func:`integrate_moment`.
        gl_chunk_size: Positive GL batch bound (default 4096) or ``None``,
            forwarded to :func:`integrate_moment` in serial and joblib workers.
        qmc_coupling_chunk_size: Optional positive dynamic coupling batch bound
            for vectorized QMC; see :func:`integrate_moment` for the callback
            contract. Each worker constructs its own arrays.
        n_samples: Number of Sobol samples (QMC only).
        seed: Random seed (QMC only).
        fixed_indices: Optional pinned index values for
            :meth:`~sft_wick.perturbation.DiagramTerm.build_integrand`.
        n_jobs: Number of parallel jobs.  ``1`` = sequential
            (default, safe under nested-callers), ``-1`` = all CPU
            cores.  A built-in guard falls
            back to sequential for ``len(diagram_terms) <= 2`` to
            avoid joblib's ~1 s startup overhead on trivial batches.
            Requires :mod:`joblib` when ``n_jobs != 1``.
        progress_tick: Optional ``tick(n=1)`` callable invoked after each
            diagram finishes (per batch on the parallel path).  Used by
            :meth:`~sft_wick.workflow.Expansion.sweep` to drive one bar
            across grid points × diagrams; never affects results.

    Returns:
        ``(total, details)`` where *total* is the scalar sum and
        *details* is a list of ``(estimate, error)`` per diagram.

    Raises:
        ValueError: if a term's R-propagator component indices contradict
            ``cache``'s R type: two different indices against a scalar-R
            cache (``compute_moment``'s default flags), an index-free R
            (``iso_R=True``) against a matrix-R cache, or two different
            indices on an R absorbed into an ``already_R_contracted``
            vertex.  The message names the flags that match the cache.
            Also when ``positions`` places two points of one R-connected
            direction group at different coordinates (see
            ``_require_one_position_per_group``).
    """
    gl_chunk_size = _validate_gl_chunk_size(gl_chunk_size)
    qmc_coupling_chunk_size = _validate_qmc_coupling_chunk_size(qmc_coupling_chunk_size)
    contraction_options = _execution_options(
        contraction_backend=contraction_backend,
        contraction_max_bytes=contraction_max_bytes,
    )
    if not diagram_terms:
        return (0.0, [])

    # Internal sweep-local binding, never a persistent cache.  Each entry is
    # static or None (the ordinary dynamic-coupling preparation path).
    prepared = (
        _prepared_integrands
        if _prepared_integrands is not None
        else (None,) * len(diagram_terms)
    )
    if len(prepared) != len(diagram_terms):
        raise ValueError("Prepared integrands do not match the diagram batch.")
    for dt, ig in zip(diagram_terms, prepared):
        if ig is not None and (
            ig.diagram_term is not dt or ig.fixed_indices != (fixed_indices or {})
        ):
            raise ValueError(
                "Prepared integrand has different diagram or component indices."
            )

    # Also checked per integrand; checking here first reports a mismatch
    # before any integrand is built or any joblib worker is started.
    iso_R = _cache_r_type(cache)
    if iso_R is not None:
        for dt in diagram_terms:
            problem = _r_index_mismatch(dt, iso_R)
            if problem:
                raise ValueError(problem)
    # Also checked per integrand, inside ``_resolve_group_x``; checking here
    # first reports it before any integrand is built or any joblib worker is
    # started.  Without a spatial table only a callable coupling reads a
    # group's coordinate, and ``_leg_positions`` refuses there.
    if positions and _cache_has_spatial_table(cache):
        for dt, ig in zip(diagram_terms, prepared):
            spatial = ig.spatial if ig is not None else dt.analyze_spatial()
            _require_one_position_per_group(
                spatial,
                _external_positions(spatial, positions),
                where="integrate_diagrams",
            )

    tick = progress_tick or (lambda n=1: None)

    if n_jobs == 1 or len(diagram_terms) <= 2:
        # Sequential: no overhead
        details = []
        for dt, ig in zip(diagram_terms, prepared):
            if ig is None:
                ig = _build_execution_integrand(
                    dt, coupling_values, fixed_indices,
                    contraction_backend, contraction_max_bytes,
                )
            val, err = integrate_moment(
                ig,
                lambda_f,
                cache,
                t_min=t_min,
                direction=direction,
                method=method,
                n_samples=n_samples,
                seed=seed,
                positions=positions,
                integrate_over=integrate_over,
                n_gauss=n_gauss,
                external_times=external_times,
                **contraction_options,
                gl_chunk_size=gl_chunk_size,
                **(
                    {}
                    if qmc_coupling_chunk_size is None
                    else {"qmc_coupling_chunk_size": qmc_coupling_chunk_size}
                ),
            )
            details.append((val, err))
            tick()
    else:
        # Parallel via joblib
        from joblib import Parallel, delayed

        # The kink probe remembers its answer on the cache, and a worker's
        # copy dies with the worker: ask here so every worker inherits it.
        if method in ("gauss_legendre", "nquad") and _c_kink_is_probed(cache):
            for dt, ig in zip(diagram_terms, prepared):
                probed = ig or _build_execution_integrand(
                    dt, coupling_values, fixed_indices,
                    contraction_backend, contraction_max_bytes,
                )
                probed._c_kink_at(
                    cache,
                    _integration_t_min(cache, t_min),
                    probed._split_externals(integrate_over, lambda_f, external_times)[2],
                    positions,
                    direction,
                )

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_eval_single_diagram)(
                dt,
                coupling_values,
                fixed_indices,
                lambda_f,
                cache,
                t_min,
                direction,
                method,
                n_samples,
                seed,
                positions,
                integrate_over,
                n_gauss,
                external_times,
                ig,
                gl_chunk_size,
                qmc_coupling_chunk_size,
                contraction_backend,
                contraction_max_bytes,
            )
            for dt, ig in zip(diagram_terms, prepared)
        )
        details = list(results)
        tick(len(details))

    total = sum(v for v, _ in details)
    return (total, details)


# ---------------------------------------------------------------------------
# Two-point correlation function integration
# ---------------------------------------------------------------------------


def _time_table_factor(
    cache: "PropagatorCache", n_l: Any, n_r: Any, t_ref: float
) -> np.ndarray | None:
    """What carries a time-only C table to the positions ``(n_l, n_r)``.

    One per component where the noise reads there as at the table's position
    (:func:`_noise_reads_alike`).  Otherwise the ratio of the diagonals of κ²
    at ``t_ref``, or ``None`` where it has no meaning: for ``C_ab`` with
    ``a != b`` (a dense R mixes the components even when κ² is diagonal), for
    σ², and where a component's κ² is 0 at ``t_ref`` and not at every time of
    the table.  A component with no noise at those times has C = 0 and takes 0.
    """
    model = cache.model
    at_table = getattr(cache, "_c_table_direction", 0.0)
    try:
        alike = _noise_reads_alike(
            model, (n_l, n_r), (at_table, at_table), cache._c_table_range
        )
    except Exception:  # noqa: BLE001 (a user callable or a stand-in)
        alike = _separation(n_l, n_r) < 1e-15
    if alike:
        return np.ones(model.n_components)
    if not getattr(model, "diag_C", True) or getattr(model, "sigma2", None) is not None:
        return None
    # A copy: a κ² may fill and return one buffer.
    diag_table = np.diag(
        np.array(_kappa2_at(model, at_table, t_ref, at_table, t_ref), dtype=float)
    )
    diag_ends = np.diag(_kappa2_at(model, n_l, t_ref, n_r, t_ref))
    zero = diag_table == 0.0
    if np.any(zero):
        lo, hi = cache._c_table_range
        for f in _NOISE_PROBE_TIMES:
            t = lo + f * (hi - lo)
            if np.any(np.diag(_kappa2_at(model, at_table, t, at_table, t))[zero]):
                return None
    return np.divide(diag_ends, diag_table, out=np.zeros_like(diag_ends), where=~zero)


def integrate_two_point_qmc(
    integrands: list["DiagramIntegrand"],
    t_f: float,
    positions: dict[str, float],
    cache: "PropagatorCache",
    t_min: float | None = None,
    n_samples: int = 2**14,
    seed: int | None = None,
    external_times: dict[str, float] | None = None,
) -> tuple[float, float]:
    """Vectorised QMC integration for two-point correlation functions.

    Evaluates the sum of diagram integrands at external time *t_f*
    with spatial points at the positions given by *positions*.  This
    is the standard entry point for computing correlators such as
    ``⟨φ(x, t) φ(y, t)⟩`` where *x* and *y* may differ.

    Unlike :func:`integrate_moment`, which assigns a single direction
    value to all spatial points, this function supports per-point
    spatial positions and applies the appropriate spatial factor to
    each correlation propagator automatically.

    The spatial factor for a *C*-propagator connecting points at
    positions *n₁* and *n₂* is computed as::

        κ²(n₁, t_ref; n₂, t_ref) / κ²(n₀, t_ref; n₀, t_ref)

    where *n₀* is the position the table was built at.  This is exact for
    separable kernels, where the factor does not depend on time.  It is not
    checked otherwise: a κ² whose time shape depends on separation gets a
    number with no warning, 57 percent low at one measured point.  Build a
    spatial table for such a kernel and call :func:`integrate_moment`.

    Requires :meth:`PropagatorCache.precompute_C_table` to have been
    called for the batch *C* evaluation fast path.

    Args:
        integrands: List of :class:`DiagramIntegrand` objects (one per
            non-vanishing Feynman diagram).
        t_f: External (observation) time: the DEFAULT time for every
            external point, and the ceiling for causal upper bounds.
        external_times: ``{point_name: time}`` overriding *t_f* per point,
            so ``C(x, t; y, t')`` and ``R(t, t')`` are reachable here too.
            ``None`` pins every external at *t_f*, bit-identically to
            before.  Same validation as
            :meth:`DiagramIntegrand.integrate_moment_qmc_vectorized`.
        positions: ``{point_name: position}`` mapping each spatial
            point name (e.g. ``"x"``, ``"y"``) to a scalar spatial
            coordinate.  Points not in the dict default to 0.
        cache: A :class:`PropagatorCache` with pre-computed *C* table.
        t_min: Lower time bound; the default is the model's ``t_min``.
        n_samples: Number of Sobol samples (should be a power of 2).
        seed: Random seed for the Sobol sequence.

    Returns:
        ``(estimate, std_error)`` for the summed two-point function.

    Raises:
        ValueError: if *positions* places two points of one R-connected
            direction group at different coordinates.  ``⟨φ(x) φ(y)⟩``
            never does (with one ψ leg per vertex no R chain joins two
            φ externals), but an observable with a response leg or a
            local vertex with two ψ legs does; see
            :func:`_require_one_position_per_group`.
    """
    from scipy.stats import qmc as _qmc

    total = 0.0
    total_err_sq = 0.0

    for ig in integrands:
        t_min = ig._admit_times(cache, t_min)
        sp = ig.spatial
        ivs = list(reversed(sp.time_integration_vars))
        evs = list(sp.external_points)

        directions = ig._resolve_group_x(
            sp, positions, 0.0, where="integrate_two_point_qmc"
        )

        # Spatial-aware dispatch, as :meth:`DiagramIntegrand._c_lookup`.  When
        # the cache has a translation / rotation / general table, ``C_at_batch``
        # evaluates C at the true endpoint positions and the kappa2 ratio below
        # must NOT also be applied, since that would double-count the
        # separation.  The ratio is the fallback for a cache that can only look
        # C up by time.
        spatial_aware = _cache_has_spatial_table(cache)
        legacy_position_blind = (
            not spatial_aware and getattr(cache, "_c_splines", None) is not None
        )

        # Spatial factor of each C propagator, by its endpoints.
        t_ref = max(t_min + 0.1, t_f * 0.5)
        c_spatial_factors = (
            {
                (sp_l, sp_r): _time_table_factor(
                    cache,
                    directions[sp.direction_map[sp_l]],
                    directions[sp.direction_map[sp_r]],
                    t_ref,
                )
                for sp_l, sp_r, _il, _ir in sp.c_propagators
            }
            if legacy_position_blind
            else {}
        )

        ni = len(ivs)
        ext_times, t_ceiling = _resolve_external_times(
            sp,
            evs,
            t_f,
            external_times,
        )

        def _scaled_C(sp_l, sp_r, t_l, t_r):
            """C from a time-only table, scaled to its separation by the
            pre-computed kappa2 ratio.  Only this lookup sees the ratio, so
            it cannot be applied on the exact path."""
            factor = c_spatial_factors.get((sp_l, sp_r))
            if legacy_position_blind and factor is None:
                raise ValueError(
                    "integrate_two_point_qmc: the cache has no spatial table, "
                    "and the kappa2 ratio that carries a time-only table to "
                    "distinct positions is defined for the diagonal components "
                    "of kappa2 only, with model.diag_C true and no sigma2, and "
                    "where kappa2 is not 0 at the reference time.  "
                    "Build a spatial table first "
                    "(precompute_C_table_translation / _rotation / _general)."
                )
            C_t = _time_table_C_batch(cache, t_l, t_r)
            if C_t.ndim == 3:
                return C_t  # the table holds this C: the factor is 1
            return C_t * factor[np.newaxis, :]

        # --- Zero integration variables ---
        #
        # ``c_spatial_factors`` (the kappa2 ratio at a single ``t_ref``)
        # exists to patch ONE blind spot: the legacy time-only spline
        # table, which ignores position entirely.  ``ig.evaluate`` ->
        # ``C_value`` is position-AWARE everywhere else: through the
        # spatial fast path when a translation/rotation/general table has
        # been built, and through the ``_C_value_direct`` (dblquad /
        # ``c_value_fn``) fallback when no table has.  In those cases the
        # delegated call is *exact* and the ratio is a strictly worse
        # approximation, so applying it there would be a regression:
        # measured 4.9% at r=0.5 rising to 34.3% at r=4 for a kernel whose
        # correlation length grows with time, where delegation reproduces
        # the closed form to every printed digit.  Routing a table-less
        # cache through the batch path is worse still, since
        # ``C_diagonal_batch`` raises, turning a working call into a
        # RuntimeError.
        #
        # So delegate whenever ``evaluate`` can see the positions, and use
        # the vectorised legacy-table-times-ratio path only when it cannot.
        # That is the *only* configuration in which the order-0 correlator
        # was ever wrong (it returned the coincident-point value at every
        # separation, a factor e^2 at r = 2 sigma).
        leg_x = (
            ig._leg_positions(positions, 0.0, where="integrate_two_point_qmc")
            if ig.dynamic_coupling is not None
            else None
        )
        if ni == 0 and not legacy_position_blind:
            et = dict(ext_times)
            ca = (
                ig.dynamic_coupling_array(et, leg_x)
                if ig.dynamic_coupling is not None
                else None
            )
            total += _real_or_raise(
                ig.evaluate(et, directions, cache, coupling_array=ca),
                ig._e_psi,
                where=" (two-point qmc, zero-dimensional)",
            )
            continue

        # --- Sobol samples (ni == 0 runs as a degenerate one-sample
        # batch with a unit Jacobian, sharing the sampled path's code so
        # the two cannot drift apart on the legacy-table convention) ---
        # bits=64: see DiagramIntegrand.integrate_moment_qmc
        u = (
            _qmc.Sobol(d=ni, seed=seed, bits=64).random(n_samples)
            if ni
            else np.empty((1, 0))
        )
        columns = {var: k for k, var in enumerate(ivs)}
        t_s, jac = _causal_times(
            _causal_bounds(sp, ivs, (), ext_times, t_f, t_ceiling, t_min), u, columns
        )
        values = ig._values_at(
            cache,
            _batch_time_reader(t_s, columns, ext_times, t_f, sp.equal_time_aliases),
            jac,
            ig._c_lookup(cache, positions, 0.0, "integrate_two_point_qmc")
            if spatial_aware
            else _scaled_C,
            leg_x,
            tag=" (dynamic coupling, two-point qmc)",
        )
        total += _finite_or_raise(float(np.mean(values)), " (two-point qmc)")

        # A zero-dimensional diagram is a single deterministic evaluation,
        # so it contributes no quadrature variance.
        if ni:
            total_err_sq += _qmc_std_error(values) ** 2

    return (total, np.sqrt(total_err_sq))
