"""Finite-window count cumulants of a linear exponential Hawkes network."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, fields, replace
from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)
from functools import cached_property, lru_cache, partial
from itertools import count, pairwise, product
from math import ceil, fsum, isfinite, log2, sqrt
from numbers import Integral
from sys import getsizeof

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import expm

from ._contraction_admission import DEFAULT_CONTRACTION_MAX_BYTES
from .workflow import (
    CustomImpulse,
    ExplicitR,
    ExponentialSpatial,
    ExponentialTemporal,
    FieldSpec,
    GaussianNoise,
    KramersMoyalVertex,
    Propagators,
    SeparableTranslation,
    System,
)


def _real(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return array.astype(float, copy=False)


def _scalar(value, name):
    array = _real(value, name)
    if array.ndim:
        raise ValueError(f"{name} must be a scalar")
    return float(array)


def _batch(function, times, shape):
    """Evaluate a scalar matrix exponential once per distinct time."""
    times = np.asarray(times)
    if not times.size:
        return np.empty(times.shape + shape)
    unique, inverse = np.unique(times.ravel(), return_inverse=True)
    return np.stack([function(float(t)) for t in unique])[inverse].reshape(
        times.shape + shape
    )


def _sum_endpoints(values):
    eps = 8 * np.finfo(float).eps
    if any(isinstance(value, Decimal) for value in values):
        endpoints = tuple(
            value if isinstance(value, Decimal) else Decimal.from_float(value)
            for value in values
        )
        total = float(sum(endpoints))
        rounding = float(Decimal.from_float(eps) * sum(map(abs, endpoints)))
    else:
        total = fsum(values)
        rounding = fsum(eps * abs(value) for value in values)
    tolerance = 5e-11 * max(1.0, abs(total))
    if not np.isfinite(total) or rounding > tolerance:
        raise ArithmeticError(
            "count-window differences exceed floating-point precision"
        )
    return total, tolerance


def _reachable(excitation, seeds):
    """Nodes reachable along positive target-row, source-column entries."""
    reached = set(seeds)
    pending = list(reached)
    while pending:
        new = set(np.flatnonzero(excitation[:, pending.pop()])) - reached
        reached.update(new)
        pending.extend(new)
    return reached


def _jump_tensor(jumps, moments, order):
    """Per-source E[jump^⊗order], counting only marked excitation legs."""
    n_nodes, dimension = jumps.shape
    is_excitation = (np.arange(dimension) < n_nodes).astype(int)
    powers = np.zeros((dimension,) * order, dtype=int)
    tensor = np.ones((n_nodes, *(dimension,) * order))
    for axis in range(order):
        shape = [1] * order
        shape[axis] = dimension
        tensor *= jumps.reshape(n_nodes, *shape)
        powers += is_excitation.reshape(shape)
    return tensor * moments[powers]


_MEAN_CONTEXT = Context(
    prec=40,
    rounding=ROUND_HALF_EVEN,
    Emin=-999999,
    Emax=999999,
    capitals=1,
    clamp=0,
    flags=[],
    traps=[InvalidOperation, DivisionByZero, Overflow],
)
_as_decimal = np.frompyfunc(Decimal.from_float, 1, 1)
# Fixed Python/NumPy allocation overhead matters for one- and two-node solvers.
_MEAN_WORKSPACE_OVERHEAD = 8 * 1024


def _mean_generator(excitation, decay, mark):
    # In coordinates (constant baseline, mean rate, mean count),
    # rate' = decay * baseline + (E[M] A - decay) rate, count' = rate.
    n = len(decay)
    gen = np.zeros((3 * n, 3 * n), dtype=excitation.dtype)
    gen[n : 2 * n, :n] = np.diag(decay)
    gen[n : 2 * n, n : 2 * n] = mark * excitation - np.diag(decay)
    gen[2 * n :, n : 2 * n] = np.eye(n, dtype=int)
    return gen


def _short_mean(gen, initial, lag):
    """Apply the exponential as a nonnegative series on the seeded state."""
    kernel = lag * gen
    shift = max(0.0, -kernel.diagonal().min())
    kernel.flat[:: len(kernel) + 1] += shift
    # Terms following a long path can be much smaller than the source.
    # Accumulate until each component converges, including the last node.
    term = state = initial
    for order in count(1):
        term = (kernel @ term) / order
        updated = state + term
        if not np.isfinite(updated).all():
            return None
        if np.array_equal(state, updated) and order >= len(initial) // 3 + 2:
            return np.exp(-shift) * updated
        state = updated


def _wide_mean(excitation, decay, mark, initial, lag, admit):
    """Scale and square exp(hG) - I with enough range for the actual immigration."""
    with localcontext(_MEAN_CONTEXT) as ctx:
        time = Decimal.from_float(lag)
        mark = Decimal.from_float(mark)
        dimension = len(initial)
        bound = max(
            1,
            Decimal.from_float(max(decay)),
            mark * Decimal.from_float(excitation.max()),
        )
        magnitude = abs(time) * (2 * dimension * bound)
        digits = max(0, magnitude.adjusted() + 1)
        # Each doubling can amplify a slow-mode rounding error. Retain the
        # extra digits implied by the time scale, independently of the caller.
        ctx.prec += digits
        cell_bytes = (
            getsizeof(Decimal((0, (9,) * ctx.prec, 0))) + np.dtype(object).itemsize
        )
        admit(
            _MEAN_WORKSPACE_OVERHEAD + 8 * cell_bytes * dimension**2,
            "wide first-moment matrix-exponential state",
        )
        scale = ceil(digits * log2(10))
        gen = _mean_generator(_as_decimal(excitation), _as_decimal(decay), mark)
        small = gen * (time / (1 << scale))
        # Summing the Taylor series without the identity keeps slow changes
        # even when a much faster node determines the first time step.
        term = delta = small
        for order in count(2):
            term = (term @ small) / order
            updated = delta + term
            if np.array_equal(updated, delta):
                break
            delta = updated
        identity = np.eye(dimension, dtype=int)
        try:
            for _ in range(scale):
                delta = delta + delta @ (identity + delta)
            return (identity + delta) @ _as_decimal(initial)
        except Overflow as exc:
            raise ArithmeticError(
                "Hawkes first moment exceeds floating-point range"
            ) from exc


def _mean_at(model):
    """Compile and cache the closed first-moment evolution on seeded paths."""
    mu = np.asarray(model.baseline)
    excitation = np.asarray(model.excitation)
    sources = np.flatnonzero(mu)
    mark = model.mark_moments[0]
    active = sorted(_reachable(excitation, sources)) if mark else list(sources)
    n = len(active)
    if not n:
        return lambda time: np.zeros((2, mu.size))
    model._admit(
        _MEAN_WORKSPACE_OVERHEAD + 10 * 8 * (3 * n) ** 2,
        "first-moment matrix-exponential state",
    )
    a = excitation[np.ix_(active, active)]
    decay = np.asarray(model.decay)[active]
    initial = np.concatenate((mu[active], mu[active], np.zeros(n)))
    with np.errstate(over="ignore", under="ignore"):
        gen = _mean_generator(a, decay, mark)
        norm = float(np.linalg.norm(gen, ord=np.inf))
        resolved_edges = not mark or np.all(
            (a == 0) | (mark * a >= np.finfo(float).tiny)
        )
    smallest = float(np.min(np.abs(gen), where=gen != 0, initial=np.inf))
    # A path to a count has at most n+2 legs before revisiting a node.
    # Keep its unit-response product away from underflow before applying mu.
    depth = n + 2
    safe_factor = depth * np.finfo(float).tiny ** (1 / depth)

    @lru_cache(maxsize=256)
    def at(time):
        if time == model.t_min:
            return np.stack((mu, np.zeros_like(mu)))
        lag = time - model.t_min
        state = None
        if (
            resolved_edges
            and abs(lag) * norm <= 8
            and abs(lag) * smallest >= safe_factor
        ):
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                state = _short_mean(gen, initial, lag)
        if state is None or (lag > 0 and np.any(state[n:] < np.finfo(float).tiny)):
            state = _wide_mean(a, decay, mark, initial, lag, model._admit)
        result = np.zeros((2, mu.size), dtype=state.dtype)
        result[:, active] = state[n:].reshape(2, n)
        return result

    return at


class _Dynamics:
    """Affine first moments, retarded response, and centred two-time covariance."""

    def __init__(self, model):
        self.model = model
        self.mu = np.asarray(model.baseline)
        n = self.mu.size
        self.dimension = d = 2 * n
        self.origin = model.t_min
        # gen, Kronecker construction, and dense expm temporaries.
        model._admit(10 * 8 * (d * d + d + 1) ** 2, "matrix-exponential state")
        a = np.asarray(model.excitation)
        self.jumps = np.concatenate((a.T, np.eye(n)), axis=1)
        self.moments = np.array((1.0, *model.mark_moments))
        self.jump2 = _jump_tensor(self.jumps, self.moments, 2)
        expected_excitation = self.moments[1] * a
        drift = np.zeros((d, d))
        drift[:n, :n] = expected_excitation - np.diag(model.decay)
        drift[n:, :n] = np.eye(n)
        # E[lambda] = mu + E[excitation]. Thus C' = KC + CKᵀ +
        # sum_j E[lambda_j] E[v_j v_jᵀ], for joint excitation/count jump v_j.
        # Propagate (vec C, E[excitation and counts], 1) from (0, 0, 1).
        gen = np.zeros((d * d + d + 1, d * d + d + 1))
        gen[: d * d, : d * d] = np.kron(drift, np.eye(d)) + np.kron(np.eye(d), drift)
        injection = self.jump2.reshape(n, d * d).T
        gen[: d * d, d * d : d * d + n] = injection
        gen[: d * d, -1] = injection @ self.mu
        gen[d * d : -1, d * d : -1] = drift
        gen[d * d : -1, -1] = np.concatenate((expected_excitation @ self.mu, self.mu))
        # These caches belong to one model and are released with it.
        self.transition = lru_cache(maxsize=256)(lambda lag: expm(lag * drift))
        self.equal_covariance = lru_cache(maxsize=256)(
            lambda t: expm((t - self.origin) * gen)[: d * d, -1].reshape(d, d).copy()
        )

    def __reduce__(self):
        # Each subprocess rebuilds the compiled matrices and local LRU caches.
        return type(self), (self.model,)

    def mean_jump(self, jumps, positions, times):
        return np.tensordot(self.rate(times[0]), jumps, axes=(-1, 0))

    def response(self, t, s):
        d = self.dimension
        return _batch(self.transition, np.asarray(t) - np.asarray(s), (d, d))

    def rate(self, t):
        rate = _batch(self.model._mean, t, (2, self.mu.size))[..., 0, :].astype(float)
        if not np.isfinite(rate).all():
            raise ArithmeticError("mean event rate exceeds floating-point range")
        return rate

    def impulse(self, x, t, y):
        return np.tensordot(self.rate(t), self.jump2, axes=(-1, 0))

    def covariance(self, x, t, y, s):
        d = self.dimension
        earlier = np.minimum(t, s)
        empty_history = earlier <= self.origin
        # Use the zero initial state for both endpoints if either has no history.
        later = np.where(empty_history, self.origin, np.maximum(t, s))
        earlier = np.where(empty_history, self.origin, earlier)
        equal_time = _batch(self.equal_covariance, earlier, (d, d))
        cross = self.response(later, earlier) @ equal_time
        return np.where(
            (np.asarray(t) >= s)[..., None, None], cross, cross.swapaxes(-1, -2)
        )

    covariance.has_diagonal_kink = True


@dataclass(frozen=True, eq=False)
class LinearHawkes:
    """A marked Hawkes network with exponential decay and empty initial history.

    An event at node j lifts excitation i by `excitation[i,j] * M`, using
    one iid non-negative mark M shared across its target nodes. Only the jth
    counter jumps, by one. The intensity is baseline plus excitation, and
    excitation i decays at rate `decay[i]`. Times are absolute from `t_min`.

    Supply one baseline per node, a square non-negative excitation matrix
    (target rows, source columns), and a positive decay per node or a common
    scalar decay. A one-node model accepts scalar parameters. The sequence
    `mark_moments` is E[M], E[M²], ..., from a common mark law. Its length
    sets the maximum KM derivative order. The default gives unit marks.

    The finite-horizon solution uses a dense state of twice the node count.
    A derivative of order m carries a dense jump tensor of rank m. Supplying
    only the moments needed by an observable controls the construction cost.
    """

    baseline: ArrayLike
    excitation: ArrayLike
    decay: ArrayLike
    mark_moments: ArrayLike = (1.0, 1.0, 1.0)
    t_min: float = 0.0
    workspace_max_bytes: int = field(
        default=DEFAULT_CONTRACTION_MAX_BYTES, kw_only=True
    )

    def __post_init__(self):
        mu = np.atleast_1d(_real(self.baseline, "baseline"))
        if mu.ndim != 1 or not mu.size or np.any(mu < 0):
            raise ValueError("baseline must be a nonempty vector of non-negative rates")
        a = _real(self.excitation, "excitation")
        if a.ndim == 0 and mu.size == 1:
            a = a.reshape(1, 1)
        if a.shape != (mu.size, mu.size) or np.any(a < 0):
            raise ValueError(
                "excitation must be a non-negative square matrix matching baseline"
            )
        decay = _real(self.decay, "decay")
        if not decay.ndim:
            decay = np.full(mu.size, float(decay))
        if decay.shape != mu.shape or np.any(decay <= 0):
            raise ValueError(
                "decay must have one positive rate per node or be a positive scalar"
            )
        moments = _real(self.mark_moments, "mark_moments")
        if moments.ndim != 1 or moments.size < 2 or np.any(moments < 0):
            raise ValueError(
                "mark_moments must give at least two non-negative raw moments"
            )
        if moments[0] == 0:
            if np.any(moments):
                raise ValueError(
                    "a zero-mean non-negative mark must have all mark_moments zero"
                )
        else:
            if np.any(moments == 0):
                raise ValueError("positive mean requires positive higher mark_moments")
            logs = np.log(np.r_[1.0, moments])
            tol = 32 * np.finfo(float).eps * np.maximum(1.0, np.abs(logs)).max()
            if np.any(np.diff(logs, n=2) < -tol):
                raise ValueError("non-negative mark_moments must be log-convex")
            if moments[1] <= float(moments[0]) * float(moments[0]):
                if np.any(np.abs(logs - np.arange(logs.size) * logs[1]) > tol):
                    raise ValueError(
                        "zero-variance mark_moments must be powers of the mean"
                    )
        if (
            isinstance(self.workspace_max_bytes, bool)
            or not isinstance(self.workspace_max_bytes, Integral)
            or self.workspace_max_bytes <= 0
        ):
            raise ValueError("workspace_max_bytes must be a positive integer")
        object.__setattr__(self, "baseline", tuple(mu.tolist()))
        object.__setattr__(self, "excitation", tuple(tuple(row) for row in a.tolist()))
        object.__setattr__(self, "decay", tuple(decay.tolist()))
        object.__setattr__(self, "mark_moments", tuple(moments.tolist()))
        object.__setattr__(self, "t_min", _scalar(self.t_min, "t_min"))

    def __getstate__(self):
        return {item.name: getattr(self, item.name) for item in fields(self)}

    def _admit(self, size, stage):
        if size > self.workspace_max_bytes:
            raise ValueError(
                f"{stage} exceeds workspace_max_bytes={self.workspace_max_bytes}; "
                "use fewer nodes or lower jump order, or raise the budget"
            )

    @cached_property
    def _mean(self):
        return _mean_at(self)

    @cached_property
    def _free(self):
        return _Dynamics(self)

    def _vertices(self, order):
        free = self._free
        jumps = (
            free.jump2 if order == 2 else _jump_tensor(free.jumps, free.moments, order)
        )
        if order > 2:
            yield KramersMoyalVertex(
                f"K{order}mean",
                order,
                partial(free.mean_jump, jumps),
                rank=order,
                coupling_vectorized=True,
            )
        # The excitation fluctuation is the fluctuation of the event rate.
        linear = np.zeros((*jumps.shape[1:], free.dimension))
        linear[..., : free.mu.size] = np.moveaxis(jumps, 0, -1)
        yield KramersMoyalVertex(f"K{order}rate", order, linear)

    @cached_property
    def _compiled(self):
        # At most the order-two, order-three and full models are requested.
        return lru_cache(maxsize=3)(self._build_system)

    @cached_property
    def _elapsed_model(self):
        return replace(self, t_min=0.0)

    @cached_property
    def _count_projection(self):
        return lru_cache(maxsize=3)(self._project_counts)

    def _project_counts(self, requested):
        """Restrict empty-history counts to paths from immigrants to readouts."""
        excitation = np.asarray(self.excitation)
        immigrants = set(np.flatnonzero(self.baseline))
        if not immigrants:
            return None, {}
        ancestry = [_reachable(excitation.T, (node,)) for node in requested]
        # An immigrant cluster must be able to contribute to every counter
        # for its product of counts to enter the joint cumulant.
        shared_sources = immigrants.intersection(*ancestry)
        if not shared_sources:
            return None, {}
        active = _reachable(excitation, shared_sources)
        retained = sorted(active.intersection(set().union(*ancestry)))
        if len(retained) == len(self.baseline) and shared_sources == immigrants:
            return self, {node: node for node in requested}
        model = replace(
            self,
            baseline=[
                self.baseline[node] if node in shared_sources else 0.0
                for node in retained
            ],
            decay=np.asarray(self.decay)[retained],
            excitation=excitation[np.ix_(retained, retained)],
        )
        return model, {node: new for new, node in enumerate(retained)}

    @property
    def system(self) -> System:
        """Centred excitation and counts as a Wick System with affine KM vertices.

        The mean-rate quadratic term is the Gaussian white covariance. All
        higher terms are tagged K<m>mean and the rate-fluctuation terms K<m>rate.
        """
        return self._compiled(len(self.mark_moments))

    def _build_system(self, order):
        d = 2 * len(self.baseline)
        # A static affine vertex is retained as nested Python floats and
        # tuples. Include those objects, its dense array and the mark tensors.
        # Checking successively also bounds arithmetic for huge requested ranks.
        size = 0
        for m in range(2, order + 1):
            size += 96 * d ** (m + 1)
            self._admit(size, "dense jump tensors")
        free = self._free
        noise = GaussianNoise(
            SeparableTranslation(ExponentialTemporal(0, 1), ExponentialSpatial(1)),
            sigma2=CustomImpulse(free.impulse),
        )
        return System(
            FieldSpec("phi", n_components=free.dimension),
            ExplicitR(free.response, iso_R=False, vectorized=True),
            noise,
            t_min=self.t_min,
            km_vertices=tuple(
                vertex for m in range(2, order + 1) for vertex in self._vertices(m)
            ),
        )

    def propagators(self, t_max: float) -> Propagators:
        """Use the model's exact centred matrix covariance for Wick evaluation."""
        return self._propagators(self.system, t_max)

    def _propagators(self, system, t_max):
        return system.propagators(
            t_max=t_max,
            c_closed_form=self._free.covariance,
            c_closed_form_vectorized=True,
            c_closed_form_only=True,
            diag_C=False,
        )

    def count_cumulant(
        self, windows: Iterable[tuple[int, float, float]], *, n_gauss: int | None = None
    ) -> float:
        """Joint cumulant of one, two or three (node, start, stop) count windows.

        Overlapping or identical windows are allowed. Two- and three-window
        results use connected Wick diagrams. By default, third-order quadrature
        doubles its Gauss–Legendre node count until the endpoint contributions
        converge. Supply n_gauss to use a specific fixed rule. Absolute window
        times are translated to the elapsed clock for the Wick calculation.
        Only immigrant clusters that can reach every counted node and the
        excitation paths by which they can affect those counts contribute.
        """
        windows = tuple(windows)
        rank = len(windows)
        if not 1 <= rank <= 3:
            raise ValueError("count_cumulant needs one, two or three windows")
        if rank > len(self.mark_moments):
            raise ValueError(f"mark_moments must reach order {rank} for this cumulant")
        if n_gauss is not None and (
            isinstance(n_gauss, bool)
            or not isinstance(n_gauss, Integral)
            or n_gauss < 1
        ):
            raise ValueError("n_gauss must be a positive integer")
        bounds, window_nodes = [], []
        n = len(self.baseline)
        for node, start, stop in windows:
            if (
                isinstance(node, bool)
                or not isinstance(node, Integral)
                or not 0 <= node < n
            ):
                raise ValueError(f"window node must be an integer in [0, {n})")
            start, stop = _scalar(start, "window start"), _scalar(stop, "window stop")
            if not self.t_min <= start <= stop:
                raise ValueError("window times must satisfy t_min <= start <= stop")
            bounds.append((start, stop))
            window_nodes.append(int(node))
        if any(a == b for a, b in bounds):
            return 0.0

        model = self
        if self.mark_moments[0] and np.any(self.excitation):
            model, remap = self._count_projection(tuple(sorted(set(window_nodes))))
            if model is None:
                return 0.0
            window_nodes = [remap[node] for node in window_nodes]

        if not model.mark_moments[0] or not np.any(model.excitation):
            # Independent Poisson counters: only the common part of windows
            # on one node contributes. Differences of large totals lose this width.
            if len(set(window_nodes)) > 1:
                return 0.0
            rate = model.baseline[window_nodes[0]]
            if rate == 0.0:
                return 0.0
            start = max(a for a, _ in bounds)
            stop = min(b for _, b in bounds)
            width = max(0.0, stop - start)
            # Scaling first avoids overflowing a width that straddles zero
            # when the final cumulant is representable.
            total = rate * width if isfinite(width) else rate * stop - rate * start
            if not isfinite(total):
                raise ArithmeticError(
                    "Poisson count cumulant exceeds floating-point range"
                )
            return total

        indices = tuple(len(model.baseline) + node for node in window_nodes)
        if model.t_min:
            # Form quadrature nodes after removing the absolute epoch. Using
            # absolute nodes would quantize the sampled elapsed times.
            times = sorted({model.t_min}.union(*bounds))
            elapsed = [time - model.t_min for time in times]
            if not isfinite(elapsed[-1]) or any(a >= b for a, b in pairwise(elapsed)):
                raise ArithmeticError(
                    "elapsed count windows exceed floating-point precision"
                )
            relative = dict(zip(times, elapsed))
            bounds = [(relative[a], relative[b]) for a, b in bounds]
            model = model._elapsed_model
        if rank == 1:
            node = window_nodes[0]
            a, b = bounds[0]
            with localcontext(_MEAN_CONTEXT):
                endpoints = (model._mean(b)[1, node], -model._mean(a)[1, node])
                return _sum_endpoints(endpoints)[0]

        labels = "xyz"[:rank]
        orders = [rank - 2]
        system = model._compiled(rank)
        expansion = system.expand(
            [f"phi_{c}({x})" for c, x in zip("abc", labels)],
            orders=orders,
            diag_R=False,
            diag_C=False,
            routing="multiplicity",
            target="connected",
        )
        horizon = max(b for _, b in bounds)
        props = model._propagators(system, horizon)

        def endpoints(nodes):
            return _count_endpoints(
                expansion, props, labels, indices, bounds, orders, nodes
            )

        if rank == 2 or n_gauss is not None:
            return _sum_endpoints(endpoints(12 if n_gauss is None else int(n_gauss)))[0]
        # The closest GL node is O(interval / nodes²) from an endpoint.
        # Start with nodes on the fastest exponential scale of the drift.
        speed = max(
            np.asarray(model.decay)
            + model.mark_moments[0] * np.sum(model.excitation, axis=1)
        )
        initial = horizon * speed
        limit = 384
        if not np.isfinite(initial) or initial > (limit // 2) ** 2:
            raise ValueError(
                "the time range needs a finer count quadrature; set n_gauss explicitly"
            )
        nodes = max(12, ceil(sqrt(initial)))
        previous = endpoints(nodes)
        while nodes < limit:
            nodes = min(limit, 2 * nodes)
            current = endpoints(nodes)
            total, tolerance = _sum_endpoints(current)
            # Compare all endpoints so opposite signs cannot hide a bad integral.
            if fsum(abs(a - b) for a, b in zip(current, previous)) <= tolerance:
                return total
            previous = current
        raise ArithmeticError(
            "count quadrature did not converge; set n_gauss explicitly"
        )


def _count_endpoints(expansion, props, labels, indices, bounds, orders, nodes):
    rank = len(bounds)
    values = []
    for bits in product((0, 1), repeat=rank):
        times = [window[bit] for window, bit in zip(bounds, bits)]
        if 0.0 in times:  # the cumulative count starts at zero
            continue
        term = expansion.evaluate(
            props,
            method="gauss_legendre",
            n_gauss=nodes,
            orders=orders,
            t_final=max(times),
            positions=dict.fromkeys(labels, 0.0),
            component_pair=tuple(indices),
            external_times=dict(zip(labels, times)),
            contraction_backend="factorized",
        )
        values.append((-1) ** (rank - sum(bits)) * term.total)
    return values
