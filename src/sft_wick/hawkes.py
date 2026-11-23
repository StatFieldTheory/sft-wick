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


def _unresolved_third(endpoints):
    """Can binary64 endpoint subtraction retain the third cumulant itself?"""
    scale = max(map(abs, endpoints))
    if not isfinite(scale):
        return True
    if not scale:
        return True
    scaled = [value / scale for value in endpoints]
    return 8 * np.finfo(float).eps * fsum(map(abs, scaled)) > 5e-11 * fsum(scaled)


def _reachable(excitation, seeds):
    """Nodes reachable along positive target-row, source-column entries."""
    reached = set(seeds)
    pending = list(reached)
    while pending:
        new = set(np.flatnonzero(excitation[:, pending.pop()])) - reached
        reached.update(new)
        pending.extend(new)
    return reached


class _MarkedJumps:
    """Per-source raw jump tensors with a separate exponent until final readout."""

    def __init__(self, jumps, moments, order):
        n_nodes, dimension = jumps.shape
        is_excitation = np.arange(dimension) < n_nodes
        powers = np.zeros((dimension,) * order, dtype=int)
        jump_fraction, jump_exponent = np.frexp(jumps)
        mark_fraction, mark_exponent = np.frexp(moments)
        tensor = np.ones((n_nodes, *(dimension,) * order))
        exponent = np.zeros(tensor.shape, dtype=jump_exponent.dtype)
        for axis in range(order):
            shape = [1] * order
            shape[axis] = dimension
            tensor *= jump_fraction.reshape(n_nodes, *shape)
            exponent += jump_exponent.reshape(n_nodes, *shape)
            powers += is_excitation.reshape(shape)
        tensor *= mark_fraction[powers]
        self.fraction, shift = np.frexp(tensor)
        self.exponent = exponent + mark_exponent[powers] + shift
        with np.errstate(over="ignore", under="ignore"):
            self.values = np.ldexp(self.fraction, self.exponent)
        self.needs_wide = bool(
            np.any(
                (self.fraction != 0)
                & ((self.values < np.finfo(float).tiny) | ~np.isfinite(self.values))
            )
        )
        self.min_exponent = int(
            np.min(
                self.exponent,
                where=self.fraction != 0,
                initial=np.iinfo(exponent.dtype).max,
            )
        )

    def finite_values(self):
        if not np.isfinite(self.values).all():
            raise ArithmeticError("marked jump tensor exceeds floating-point range")
        return self.values

    def weighted(self, rates):
        """Sum rate times jump moment, rounding after combining both factors."""
        rates = np.asarray(rates)
        numeric = rates.astype(float, copy=False)
        if not np.isfinite(numeric).all():
            raise ArithmeticError("mean event rate exceeds floating-point range")
        smallest = np.min(np.abs(numeric), where=rates.astype(bool), initial=np.inf)
        # Two frexp fractions multiply to at least 1/4. If every nonzero
        # product is normal the usual BLAS sum can use the rounded tensor.
        if (
            not self.needs_wide
            and smallest >= np.finfo(float).tiny
            and self.min_exponent + int(np.frexp(smallest)[1])
            >= np.finfo(float).minexp + 2
        ):
            with np.errstate(over="ignore", invalid="ignore"):
                result = np.tensordot(numeric, self.values, axes=(-1, 0))
        else:
            result = np.empty(rates.shape[:-1] + self.values.shape[1:])
            with (
                localcontext(_MEAN_CONTEXT),
                np.errstate(over="ignore", under="ignore"),
            ):
                for index in np.ndindex(rates.shape[:-1]):
                    result[index] = self._weighted_row(rates[index])
        if not np.isfinite(result).all():
            raise ArithmeticError(
                "weighted marked jump tensor exceeds floating-point range"
            )
        return result

    def _weighted_row(self, rates):
        shape = (-1,) + (1,) * (self.fraction.ndim - 1)
        fractions, powers = np.frexp(rates.astype(float))
        # A wide mean can be smaller than the least binary64 subnormal.
        # Move that rate near one before converting its significand to float.
        if rates.dtype == object:
            for j, rate in enumerate(rates):
                if (
                    isinstance(rate, Decimal)
                    and rate
                    and abs(float(rate)) < np.finfo(float).tiny
                ):
                    power = int(rate.adjusted() * log2(10))
                    frac, shift = np.frexp(float(rate / (Decimal(2) ** power)))
                    fractions[j], powers[j] = frac, power + int(shift)
        terms = self.fraction * fractions.reshape(shape)
        exponents = self.exponent + powers.reshape(shape)
        missing = np.iinfo(exponents.dtype).min
        largest = np.max(exponents, axis=0, where=terms != 0, initial=missing)
        largest = np.where(largest == missing, 0, largest)
        scaled_sum = np.sum(np.ldexp(terms, exponents - largest), axis=0)
        return np.ldexp(scaled_sum, largest)


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


def _excitation_drift(a, decay, mark):
    drift = mark * a - np.diag(decay)
    if drift.dtype != object:
        # A product rounded before subtracting decay can erase the entire
        # residual of a near-critical node. Only the diagonal subtracts.
        with localcontext(_MEAN_CONTEXT) as ctx:
            ctx.prec = 80
            for i, rate in enumerate(decay):
                value = Decimal.from_float(mark).fma(
                    Decimal.from_float(a[i, i]), -Decimal.from_float(rate)
                )
                drift[i, i] = float(value)
    return drift


def _mean_generator(excitation, decay, mark):
    # In coordinates (constant baseline, mean rate, mean count),
    # rate' = decay * baseline + (E[M] A - decay) rate, count' = rate.
    n = len(decay)
    gen = np.zeros((3 * n, 3 * n), dtype=excitation.dtype)
    gen[n : 2 * n, :n] = np.diag(decay)
    gen[n : 2 * n, n : 2 * n] = _excitation_drift(excitation, decay, mark)
    gen[2 * n :, n : 2 * n] = np.eye(n, dtype=int)
    return gen


def _short_action(gen, initial, lag, depth):
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
        if np.array_equal(state, updated) and order >= depth:
            return np.exp(-shift) * updated
        state = updated


def _wide_action(generator, initial, lag, bound, admit, stage):
    """Apply a scaled matrix exponential while retaining the physical range."""
    with localcontext(_MEAN_CONTEXT) as ctx:
        time = Decimal.from_float(lag)
        dimension = len(initial)
        bound = bound()
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
            f"wide {stage} matrix-exponential state",
        )
        scale = ceil(digits * log2(10))
        gen = generator()
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
            # Square the transition itself, so a decaying mode remains
            # visible even when exp(hG)-I has rounded to -1 on its diagonal.
            transition = identity + delta
            for _ in range(scale):
                transition = transition @ transition
            state = initial if initial.dtype == object else _as_decimal(initial)
            return transition @ state
        except Overflow as exc:
            raise ArithmeticError(f"Hawkes {stage} exceeds numerical range") from exc


def _wide_mean(excitation, decay, mark, initial, lag, admit):
    def generator():
        return _mean_generator(
            _as_decimal(excitation), _as_decimal(decay), Decimal.from_float(mark)
        )

    def bound():
        return max(
            1,
            Decimal.from_float(max(decay)),
            Decimal.from_float(mark) * Decimal.from_float(excitation.max()),
        )

    return _wide_action(generator, initial, lag, bound, admit, "first-moment")


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
        resolved_edges = _resolved_edges(a, mark)
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
                state = _short_action(gen, initial, lag, depth)
        if state is None or (lag > 0 and np.any(state[n:] < np.finfo(float).tiny)):
            state = _wide_mean(a, decay, mark, initial, lag, model._admit)
        result = np.zeros((2, mu.size), dtype=state.dtype)
        result[:, active] = state[n:].reshape(2, n)
        return result

    return at


def _resolved_action(gen, seed, result, floor_bits):
    """Every reachable output must exceed the accumulated underflow allowance."""
    weak = np.frexp(result)[1] <= floor_bits + 1
    weak |= result == 0
    if not np.any(weak):
        return True
    # An unreachable coordinate stays exactly zero at every precision.
    support = seed != 0
    edges = gen > 0
    while not np.any(weak & support):
        following = support | (edges @ support)
        if np.array_equal(following, support):
            return True
        support = following
    return False


def _linear_flow(
    gen, wide_generator, bound, depth, admit, stage, resolved, seed_size=0, damping=0.0
):
    """Return the action of a constant Metzler generator on a vector or matrix."""
    with np.errstate(over="ignore"):
        norm = float(np.linalg.norm(gen, ord=np.inf))
    smallest = float(np.min(np.abs(gen), where=gen != 0, initial=1.0))

    # A zero row preserves its coordinate; a zero column preserves a unit
    # impulse there. These exact diagonal ones include constant immigration
    # and cumulative count covariances. A one-ulp error in that unit would
    # double with each squaring even after every other mode has relaxed.
    fixed = np.flatnonzero(~np.any(gen, axis=0) | ~np.any(gen, axis=1))
    # A positive damping margin bounds the lifetime of transient rounding
    # errors. Compare it with the scaling norm to exclude stiff systems.
    well_damped = damping > 0 and norm <= 256 * damping

    def apply(lag, state):
        if lag == 0 or not np.any(state):
            return state
        magnitude = abs(lag) * norm
        # A short nonnegative series retains small causal paths. Limit
        # binary64 squarings so accumulated relative rounding stays small.
        if (
            resolved
            and state.dtype != object
            and isfinite(magnitude)
            and (magnitude <= 4096 or well_damped)
        ):
            squares = max(0, ceil(log2(max(1.0, magnitude / 8))))
            step_lag = lag / (1 << squares)
            factor = min(1.0, abs(step_lag) * smallest / depth)
            seed = float(np.min(np.abs(state), where=state != 0, initial=1.0))
            # Keep the weakest seeded path in range, including the diagonal
            # attenuation. Otherwise preserve it for subsequent propagation.
            attenuation = 2 * abs(lag) * max(0.0, -float(np.min(gen.diagonal())))
            if seed_size and not np.any(state[seed_size:]):
                # Empty-history covariance gains independent immigrant
                # clusters with time. Its seeded paths cannot decay away.
                attenuation = 0.0
            safe_path = (
                factor and log2(seed) + depth * log2(factor) - attenuation > -1022
            )
            if safe_path or well_damped:
                # If a path bound is inconclusive, bound absolute loss from
                # subnormal products in the nonnegative transition itself.
                # Reserve 45 bits relative to each requested output.
                check_loss = not safe_path
                floor_bits = -1022 + 2 * log2(len(gen)) + 45

                with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                    if squares:
                        step = _short_action(gen, np.eye(len(gen)), step_lag, depth)
                        step[fixed, fixed] = 1.0
                        for _ in range(squares):
                            if check_loss:
                                # An error E in P is amplified by at most
                                # 2||P|| + ||E|| when P is squared.
                                floor_bits += log2(2 * np.linalg.norm(step, np.inf) + 2)
                            step = step @ step
                        result = step @ state
                    else:
                        result = _short_action(gen, state, lag, depth)
                if result is not None and np.isfinite(result).all():
                    if not check_loss or _resolved_action(
                        gen, state, result, floor_bits + max(0.0, log2(np.max(state)))
                    ):
                        return result
        return _wide_action(wide_generator, state, lag, bound, admit, stage)

    return apply


def _drift(a, decay, mark, readout):
    n = len(decay)
    drift = np.zeros((n + len(readout),) * 2, dtype=a.dtype)
    drift[:n, :n] = _excitation_drift(a, decay, mark)
    drift[n:, :n] = readout
    return drift


def _covariance_generator(drift, decay, jump2):
    # Propagate (constant baseline, mean rate, vec C). Independent clusters
    # make C linear in immigration; all baseline values belong in the seed.
    n, d = len(decay), len(drift)
    gen = np.zeros((2 * n + d * d, 2 * n + d * d), dtype=drift.dtype)
    offset = 2 * n
    gen[n:offset, :n] = np.diag(decay)
    gen[n:offset, n:offset] = drift[:n, :n]
    gen[offset:, n:offset] = jump2.reshape(n, d * d).T
    eye = np.eye(d, dtype=int)
    gen[offset:, offset:] = np.kron(drift, eye) + np.kron(eye, drift)
    return gen


def _resolved_edges(a, mark):
    with np.errstate(over="ignore", under="ignore"):
        return not mark or np.all((a == 0) | (mark * a >= np.finfo(float).tiny))


def _response_flow(a, decay, mark, admit, readout=None):
    if readout is None:
        readout = np.eye(len(decay), dtype=int)

    def wide_drift():
        return _drift(
            _as_decimal(a), _as_decimal(decay), Decimal.from_float(mark), readout
        )

    def bound():
        return max(
            1,
            Decimal.from_float(max(decay)),
            Decimal.from_float(mark) * Decimal.from_float(a.max()),
        )

    return _linear_flow(
        _drift(a, decay, mark, readout),
        wide_drift,
        bound,
        len(decay) + 1,
        admit,
        "response",
        _resolved_edges(a, mark),
    )


def _covariance_flow(model, a, decay, readout, jump2):
    mark, mark2 = model.mark_moments[:2]
    n = len(decay)
    dimension = 2 * n + (n + len(readout)) ** 2
    model._admit(10 * 8 * dimension**2, "covariance matrix-exponential state")

    def wide_generator():
        amplitude, beta = _as_decimal(a), _as_decimal(decay)
        m1, m2 = map(Decimal.from_float, (mark, mark2))
        jumps = np.concatenate((amplitude.T, readout.T), axis=1)
        second = jumps[:, :, None] * jumps[:, None, :]
        second[:, :n, :n] *= m2
        second[:, :n, n:] *= m1
        second[:, n:, :n] *= m1
        return _covariance_generator(_drift(amplitude, beta, m1, readout), beta, second)

    def bound():
        largest = Decimal.from_float(a.max())
        return 2 * max(
            1,
            Decimal.from_float(max(decay)),
            Decimal.from_float(mark) * largest,
            Decimal.from_float(mark2) * largest**2,
        )

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        gen = _covariance_generator(_drift(a, decay, mark, readout), decay, jump2)
    # Every positive pair of marked excitation legs must survive float
    # construction before a fast solve multiplies it by the immigration.
    nz = a[a > 0]
    resolved = _resolved_edges(a, mark)
    if nz.size and mark2:
        with localcontext(_MEAN_CONTEXT):
            smallest = Decimal.from_float(nz.min())
            resolved = resolved and Decimal.from_float(
                mark2
            ) * smallest**2 >= Decimal.from_float(np.finfo(float).tiny)
    # The infinity logarithmic norm of the excitation drift bounds the
    # decay of its semigroup for a Metzler matrix. The covariance adds its
    # Kronecker sum and the exactly neutral coordinates handled above.
    drift = gen[n : 2 * n, n : 2 * n]
    damping = -float(np.max(drift.sum(axis=1)))
    return _linear_flow(
        gen,
        wide_generator,
        bound,
        3 * n + 2,
        model._admit,
        "covariance",
        resolved,
        seed_size=2 * n,
        damping=damping,
    )


def _representable(array, name):
    result = np.asarray(array, dtype=float)
    if not np.isfinite(result).all():
        raise ArithmeticError(f"Hawkes {name} exceeds floating-point range")
    return result


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
        self.second_jumps = _MarkedJumps(self.jumps, self.moments, 2)
        decay = np.asarray(model.decay)
        mark = model.mark_moments[0]
        response = _response_flow(a, decay, mark, model._admit)
        eye = np.eye(d)
        self.transition = lru_cache(maxsize=256)(
            lambda lag: _representable(response(lag, eye), "response")
        )
        sources = np.flatnonzero(self.mu)
        active = sorted(_reachable(a, sources)) if mark else list(sources)
        indices = np.r_[active, np.asarray(active, dtype=int) + n].astype(int)
        if not active:
            self.cross = lambda later, earlier: np.zeros((d, d))
            return
        if len(active) != n:
            a = a[np.ix_(active, active)]
            decay = decay[active]
            response_active = _response_flow(a, decay, mark, model._admit)
        else:
            response_active = response
        jump2 = self.second_jumps.values[np.ix_(active, indices, indices)]
        evolve = _covariance_flow(
            model, a, decay, np.eye(len(active), dtype=int), jump2
        )
        size = len(indices)
        initial = np.concatenate((self.mu[active], self.mu[active], np.zeros(size**2)))

        @lru_cache(maxsize=256)
        def equal(time):
            return evolve(time - self.origin, initial)[size:].reshape(size, size)

        @lru_cache(maxsize=256)
        def cross(later, earlier):
            result = np.zeros((d, d))
            if earlier > self.origin:
                block = response_active(later - earlier, equal(earlier))
                result[np.ix_(indices, indices)] = _representable(block, "covariance")
            return result

        self.cross = cross

    def __reduce__(self):
        # Each subprocess rebuilds the compiled matrices and local LRU caches.
        return type(self), (self.model,)

    def mean_jump(self, jumps, positions, times):
        return jumps.weighted(self.rate(times[0]))

    def response(self, t, s):
        d = self.dimension
        return _batch(self.transition, np.asarray(t) - np.asarray(s), (d, d))

    def rate(self, t):
        return _batch(self.model._mean, t, (2, self.mu.size))[..., 0, :]

    def impulse(self, x, t, y):
        return self.second_jumps.weighted(self.rate(t))

    def covariance(self, x, t, y, s):
        t, s = np.broadcast_arrays(t, s)
        if not t.size:
            return np.empty(t.shape + (self.dimension, self.dimension))
        pairs = np.stack((np.maximum(t, s).ravel(), np.minimum(t, s).ravel()), axis=1)
        unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
        matrices = np.stack([self.cross(float(a), float(b)) for a, b in unique])
        cross = matrices[inverse].reshape(t.shape + (self.dimension, self.dimension))
        return np.where((t >= s)[..., None, None], cross, cross.swapaxes(-1, -2))

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
            free.second_jumps
            if order == 2
            else _MarkedJumps(free.jumps, free.moments, order)
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
        values = jumps.finite_values()
        linear = np.zeros((*values.shape[1:], free.dimension))
        linear[..., : free.mu.size] = np.moveaxis(values, 0, -1)
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

    def _window_covariance(self, nodes, windows):
        """The single connected Wick C line, with each counter open in its window."""
        n = len(self.baseline)
        a, decay = np.asarray(self.excitation), np.asarray(self.decay)
        dimension = n + 2
        size = 2 * n + dimension**2
        self._admit(10 * 8 * size**2, "covariance matrix-exponential state")
        state = np.r_[self.baseline, self.baseline, np.zeros(dimension**2)]
        frozen = min(range(2), key=lambda i: windows[i][1])
        remaining = 1 - frozen
        close = windows[frozen][1]
        times = sorted(t for t in {self.t_min}.union(*windows) if t <= close)
        moments = np.array((1.0, *self.mark_moments[:2]))
        for left, right in pairwise(times):
            readout = np.zeros((2, n), dtype=int)
            for i, (node, (start, stop)) in enumerate(zip(nodes, windows)):
                readout[i, node] = int(start <= left < stop)
            jumps = np.concatenate((a.T, readout.T), axis=1)
            jump2 = _MarkedJumps(jumps, moments, 2).values
            evolve = _covariance_flow(self, a, decay, readout, jump2)
            state = evolve(right - left, state)
        covariance = state[2 * n :].reshape(dimension, dimension)
        result = covariance[n, n + 1]
        start, stop = windows[remaining]
        if close < stop:
            # Once F closes there are no new jumps on its leg. Only
            # Cov(X_i,F) and Cov(the remaining counter,F) are needed.
            column = np.r_[covariance[:n, n + frozen], result]
            readout = np.zeros((1, n), dtype=int)
            if close < start:
                gap = _response_flow(
                    a, decay, self.mark_moments[0], self._admit, readout
                )
                column = gap(start - close, column)
            readout = np.zeros((1, n), dtype=int)
            readout[0, nodes[remaining]] = 1
            tail = _response_flow(a, decay, self.mark_moments[0], self._admit, readout)
            result = tail(stop - max(start, close), column)[-1]
        # Select before conversion: unobserved excitation variances may
        # exceed binary64 even when the requested count covariance fits.
        return float(_representable(result, "count covariance"))

    def count_cumulant(
        self, windows: Iterable[tuple[int, float, float]], *, n_gauss: int | None = None
    ) -> float:
        """Joint cumulant of one, two or three (node, start, stop) count windows.

        Overlapping or identical windows are allowed. Two-window results
        propagate the connected covariance with each counter active only in
        its window. Three-window results use connected Wick diagrams.
        By default, third-order quadrature doubles the Gauss–Legendre node
        count to check convergence. Supply n_gauss for a fixed rule. Absolute window
        times are translated to the elapsed clock for the Wick calculation.
        If diagram factors need extended range, fixed rules retain the same
        cumulative-endpoint quadrature. Adaptive rules and severely cancelling
        endpoint differences integrate directly between changes of readout.
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

        if rank == 2:
            return model._window_covariance(window_nodes, bounds)

        from ._hawkes_third import needs_wide_third, window_third

        if needs_wide_third(model, bounds):
            return window_third(model, window_nodes, bounds, n_gauss)

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
            model._admit(64 * int(nodes) ** 2, "third-cumulant quadrature")
            return _count_endpoints(
                expansion, props, labels, indices, bounds, orders, nodes
            )

        if n_gauss is not None:
            fixed = endpoints(int(n_gauss))
            if _unresolved_third(fixed):
                return window_third(model, window_nodes, bounds, n_gauss)
            return _sum_endpoints(fixed)[0]
        # The closest GL node is O(interval / nodes²) from an endpoint.
        # Start with nodes on the fastest exponential scale of the drift.
        speed = max(
            np.asarray(model.decay)
            + np.sum(model.mark_moments[0] * np.asarray(model.excitation), axis=1)
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
            if _unresolved_third(current):
                return window_third(model, window_nodes, bounds, None)
            total, _ = _sum_endpoints(current)
            tolerance = 5e-11 * abs(total)
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
