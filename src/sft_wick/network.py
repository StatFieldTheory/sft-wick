"""Stochastic reaction–diffusion on a finite graph with sites as components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import Context, Decimal, localcontext
from functools import cached_property, lru_cache
from math import ceil, log, log10, log2
from numbers import Integral

import numpy as np
from scipy.linalg import eigh, expm, solve_continuous_lyapunov
from scipy.sparse.csgraph import connected_components

from ._contraction_admission import DEFAULT_CONTRACTION_MAX_BYTES
from .simplify import _apply_index_sub, simplify
from .workflow import (
    ConstantImpulse,
    ExplicitR,
    ExponentialSpatial,
    ExponentialTemporal,
    FieldSpec,
    GaussianNoise,
    InitialState,
    LocalVertex,
    SeparableTranslation,
    System,
)
from .workflow._covariance import is_covariance
from .workflow._markov_embedding import _DrivenCovariance


def _array(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must contain finite real numbers")
    array = np.array(raw, dtype=float, copy=True)
    array.setflags(write=False)
    return array


def _scalar(value, name):
    value = _array(value, name)
    if value.ndim:
        raise ValueError(f"{name} must be scalar")
    return float(value)


def _site_indices(term):
    """Use the common node at each local vertex as its only summed index."""
    by_point = {}
    substitutions = {}
    points = set(term.integration_vars)
    for line in term.propagators:
        for label, index in (
            (line.spatial_left, line.index_left),
            (line.spatial_right, line.index_right),
        ):
            if label in points and index is not None:
                substitutions[index] = by_point.setdefault(label, index)
    return replace(
        term,
        propagators=tuple(
            _apply_index_sub(line, substitutions) for line in term.propagators
        ),
        coupling_sum=simplify(_apply_index_sub(term.coupling_sum, substitutions)),
        summation_indices=tuple(
            item
            for item in term.summation_indices
            if substitutions.get(item[0], item[0]) == item[0]
        ),
    )


_decimal_array = np.frompyfunc(Decimal.from_float, 1, 1)


@lru_cache(maxsize=32)
def _coordinate_flow(model, time, stationary=False, extra=0):
    """Response and noise Gramian on a short step, then forward doubling.

    Construct the conservative generator from its edge rates. Extra digits
    retain the damping when a site exchanges mass much faster than it decays.
    Evolving the physical covariance preserves the support of singular noise.
    """
    if not stationary and not time:
        n = len(model.noise)
        return np.eye(n, dtype=int).astype(object), np.zeros((n, n), dtype=int).astype(
            object
        )
    with localcontext(Context(prec=50 + extra)) as ctx:
        gamma = Decimal.from_float(model.gamma)
        diffusion = Decimal.from_float(model.diffusion)
        lag = Decimal(1) / gamma if stationary else Decimal.from_float(time)
        rate = gamma + 2 * diffusion * Decimal.from_float(
            float(model.laplacian.diagonal().max())
        )
        digits = max(0, (lag * rate).adjusted() + 1)
        ctx.prec += digits
        if stationary:
            lag = Decimal(1) / gamma
        squarings = ceil(digits * log2(10)) + 1
        h = lag / (1 << squarings)
        generator = -diffusion * _decimal_array(model.laplacian)
        np.fill_diagonal(generator, 0)
        np.fill_diagonal(generator, -generator.sum(axis=1) - gamma)
        step = h * generator
        n = len(step)
        transition = increment = np.eye(n, dtype=int).astype(object)
        covariance = gramian_term = h * _decimal_array(model.noise)
        # Integral_0^h exp(Au) Q exp(A^T u) du has Taylor coefficients
        # given by powers of the Lyapunov operator, B -> AB + BA^T.
        order = 0
        while True:
            order += 1
            increment = (step @ increment) / Decimal(order)
            gramian_term = (step @ gramian_term + gramian_term @ step.T) / Decimal(
                order + 1
            )
            new_transition = transition + increment
            new_covariance = covariance + gramian_term
            done = (
                order >= 2 * n
                and np.array_equal(new_transition, transition)
                and np.array_equal(new_covariance, covariance)
            )
            transition, covariance = new_transition, new_covariance
            if done:
                break
        for _ in range(squarings):
            covariance = covariance + transition @ covariance @ transition.T
            transition = transition @ transition
        if stationary:
            # After time 1/gamma the semigroup contracts in the maximum
            # norm by e^-1. Repeated doubling sums the entire remaining tail.
            while True:
                new_covariance = covariance + transition @ covariance @ transition.T
                if np.array_equal(new_covariance, covariance):
                    break
                covariance = new_covariance
                transition = transition @ transition
        return transition, covariance


def _at_lags(lags, scalar):
    lags = np.asarray(lags)
    if not lags.ndim:
        return scalar(float(lags))
    values, inverse = np.unique(lags, return_inverse=True)
    blocks = np.stack([scalar(float(t)) for t in values])
    return blocks[inverse].reshape(lags.shape + blocks.shape[-2:])


@dataclass(frozen=True, eq=False, init=False)
class NetworkSystem:
    """A matrix Ornstein–Uhlenbeck process with node-local polynomial drift.

    The linear drift is `-(gamma I + diffusion L) x`. The supplied graph
    Laplacian has zero row sums and nonpositive off-diagonals. A weight
    from source j to target i is `-L[i, j]`. The white covariance rate
    `noise` may be a scalar, a vector of variances, or a PSD matrix.
    Each entry `power: coefficients` of `local_drift` supplies the
    physical term `coefficients[i] * x[i]**power` at site i. Coefficients
    may be scalar or per-site. Perturbation orders count these vertices.
    """

    laplacian: np.ndarray
    gamma: float
    diffusion: float
    noise: np.ndarray
    t_min: float
    start: str
    local_drift: tuple

    def __init__(
        self,
        laplacian,
        gamma,
        diffusion,
        noise=1.0,
        *,
        local_drift=None,
        t_min=0.0,
        start="zero",
    ):
        laplacian = _array(laplacian, "laplacian")
        if (
            laplacian.ndim != 2
            or not len(laplacian)
            or laplacian.shape[0] != laplacian.shape[1]
        ):
            raise ValueError("laplacian must be a nonempty square matrix")
        n = len(laplacian)
        diagonal = np.diag(laplacian)
        if np.any(laplacian - np.diag(diagonal) > 0) or not np.allclose(
            laplacian.sum(axis=1), 0, rtol=0, atol=32 * np.finfo(float).eps * diagonal
        ):
            raise ValueError(
                "laplacian requires nonpositive off-diagonals and zero row sums"
            )
        gamma, diffusion = _scalar(gamma, "gamma"), _scalar(diffusion, "diffusion")
        if gamma < 0 or diffusion < 0:
            raise ValueError("gamma and diffusion must be nonnegative")
        if start not in ("zero", "stationary") or (
            start == "stationary" and gamma == 0
        ):
            raise ValueError(
                "start must be zero or stationary; stationary requires gamma > 0"
            )
        q = _array(noise, "noise")
        if q.ndim == 0:
            q = float(q) * np.eye(n)
        elif q.shape == (n,):
            q = np.diag(q)
        if q.shape != (n, n) or not np.array_equal(q, q.T) or not is_covariance(q):
            raise ValueError(
                "noise must give a symmetric positive semidefinite covariance rate"
            )
        q.setflags(write=False)
        if local_drift is not None and not isinstance(local_drift, Mapping):
            raise ValueError("local_drift must map powers to site coefficients")
        entries = []
        for power, coefficient in ({} if local_drift is None else local_drift).items():
            if isinstance(power, bool) or not isinstance(power, Integral) or power < 1:
                raise ValueError("local drift powers must be positive integers")
            coefficient = _array(coefficient, "local drift")
            if coefficient.ndim and coefficient.shape != (n,):
                raise ValueError(
                    "each local drift needs a scalar or one coefficient per site"
                )
            entries.append((int(power), np.broadcast_to(coefficient, (n,))))
        for key, val in (
            ("laplacian", laplacian),
            ("gamma", gamma),
            ("diffusion", diffusion),
            ("noise", q),
            ("start", start),
            ("t_min", _scalar(t_min, "t_min")),
            ("local_drift", tuple(entries)),
        ):
            object.__setattr__(self, key, val)

    @cached_property
    def _drift(self):
        return (
            -self.gamma * np.eye(len(self.laplacian)) - self.diffusion * self.laplacian
        )

    @cached_property
    def _modes(self):
        if not np.array_equal(self.laplacian, self.laplacian.T):
            return None
        values, vectors = eigh(self.laplacian)
        # Each connected component has an exact constant mode. Preserve its
        # lifetime also when the horizon resolves roundoff in a zero eigenvalue.
        components = connected_components(
            self.laplacian < 0, directed=False, return_labels=False
        )
        if components < len(values):
            # The backward error scales with the fastest mode. If it can
            # cost the slow contrast modes their requested accuracy, their
            # eigenvectors may also mix with consensus. Use physical
            # coordinates, whose generator is assembled from edge rates.
            relative_resolution = len(values) * np.finfo(float).eps / 1e-12
            if values[components] <= relative_resolution * values[-1]:
                return None
        values[:components] = 0.0
        rates = self.gamma + self.diffusion * np.maximum(0, values)
        return rates, vectors, vectors.T @ self.noise @ vectors

    @cached_property
    def _exchange(self):
        return self.diffusion * float(self.laplacian.diagonal().max())

    @cached_property
    def _weak_forcing(self):
        # A modal round-trip can turn a small or unforced site into signed
        # roundoff when heat exchange or forcing there is sufficiently weak.
        spectrum = np.linalg.eigvalsh(self.noise)
        return (
            spectrum[0] <= 1e-12 * spectrum[-1] or self._exchange <= 1e-5 * self.gamma
        )

    def _response(self, t, s):
        lag = np.maximum(np.subtract(t, s), 0.0)
        if not self._exchange:
            return np.exp(-self.gamma * lag)[..., None, None] * np.eye(len(self.noise))
        if self._modes is None and np.max(lag, initial=0.0) * self._exchange > 1000:
            return _at_lags(
                lag, lambda dt: np.asarray(_coordinate_flow(self, dt)[0], dtype=float)
            )
        if self._modes is None or self._exchange <= 1e-5 * self.gamma:
            return expm(lag[..., None, None] * self._drift)
        rates, vectors, _ = self._modes
        return (vectors * np.exp(-lag[..., None, None] * rates)) @ vectors.T

    @cached_property
    def _directed_covariance(self):
        return _DrivenCovariance(self._drift, self.noise, self.t_min)

    def _signed_cross(self, t, s, stationary):
        # Consensus in R can be order one while its product with a signed
        # covariance selects only a rapidly decaying contrast. Keep the
        # two factors together until the matrix product is complete.
        lag = abs(t - s)
        elapsed = max(0.0, min(t, s) - self.t_min)
        if not stationary and not elapsed:
            return np.zeros_like(self.noise)
        clock_log = -log10(self.gamma) if stationary else log10(elapsed)
        full = 350 + max(0.0, log10(float(np.abs(self.noise).max())) + clock_log)
        digits = ceil(min(full, lag * (2 * self._exchange + self.gamma) / log(10)))
        with localcontext(Context(prec=50 + digits)):
            forward = _coordinate_flow(self, lag, extra=digits)[0]
            same_time = _coordinate_flow(
                self, 0.0 if stationary else elapsed, stationary, extra=digits
            )[1]
            cross = np.asarray(forward @ same_time, dtype=float)
        return cross if t >= s else cross.T

    def _signed_covariance(self, x, t, y, s):
        t, s = np.broadcast_arrays(t, s)
        if not t.size:
            return np.empty(t.shape + self.noise.shape)
        lag = np.abs(t - s)
        if not self._weak_forcing and np.max(lag) * self._exchange < 4:
            if self.start == "zero":
                return self._driven(x, t, y, s)
            forward = self._response(np.maximum(t, s), np.minimum(t, s))
            cross = forward @ self.system.initial_state.covariance
            return np.where((t >= s)[..., None, None], cross, cross.swapaxes(-1, -2))
        pairs, inverse = np.unique(
            np.stack((t.ravel(), s.ravel()), axis=1), axis=0, return_inverse=True
        )
        matrices = [
            self._signed_cross(float(a), float(b), self.start == "stationary")
            for a, b in pairs
        ]
        return np.stack(matrices)[inverse].reshape(t.shape + self.noise.shape)

    setattr(_signed_covariance, "has_diagonal_kink", True)

    def _driven(self, x, t, y, s):
        t, s = np.broadcast_arrays(t, s)
        early = np.minimum(t, s)
        elapsed = np.maximum(0.0, early - self.t_min)
        if not self._exchange:
            # Sites have a common scalar response, including gamma=0.
            kernel = (
                elapsed
                if not self.gamma
                else -np.expm1(-2 * (self.gamma * elapsed)) / self.gamma / 2
            )
            decay = np.exp(-self.gamma * np.abs(t - s))
            return (decay * kernel)[..., None, None] * self.noise
        if not elapsed.size:
            return np.empty(t.shape + self.noise.shape)
        if self._weak_forcing or (
            self._modes is None and np.max(elapsed, initial=0.0) * self._exchange > 1000
        ):
            covariance = _at_lags(
                elapsed,
                lambda dt: np.asarray(_coordinate_flow(self, dt)[1], dtype=float),
            )
            return (
                self._response(t, early)
                @ covariance
                @ self._response(s, early).swapaxes(-1, -2)
            )
        if self._modes is None:
            # Separate unequal-time response so long lags use the same
            # conservative propagator as public R.
            covariance = self._directed_covariance._same_time(elapsed)
            return (
                self._response(t, early)
                @ covariance
                @ self._response(s, early).swapaxes(-1, -2)
            )
        elapsed = elapsed[..., None, None]
        rates, vectors, q = self._modes
        pair = rates[:, None] + rates
        factors = -np.expm1(-elapsed * pair)
        factors = np.divide(
            factors,
            pair,
            out=np.broadcast_to(elapsed, factors.shape).copy(),
            where=factors != 0,
        )
        left = np.exp(-(t - early)[..., None] * rates)
        right = np.exp(-(s - early)[..., None] * rates)
        modes = left[..., :, None] * (q * factors) * right[..., None, :]
        return vectors @ modes @ vectors.T

    setattr(_driven, "has_diagonal_kink", True)

    @cached_property
    def system(self):
        """The ordinary Wick System, including the full site tensors."""
        n = len(self.laplacian)
        vertices = []
        cells = 0
        for power, coefficients in self.local_drift:
            if power + 1 > 64:
                raise ValueError(
                    "on-site drift rank exceeds the NumPy tensor rank limit"
                )
            if n > 1 and power + 1 > (DEFAULT_CONTRACTION_MAX_BYTES // 24).bit_length():
                raise ValueError(
                    "on-site drift tensors exceed the dense coupling workspace"
                )
            elements = n ** (power + 1)
            cells += elements
            if 24 * cells > DEFAULT_CONTRACTION_MAX_BYTES:
                raise ValueError(
                    "on-site drift tensors exceed the dense coupling workspace"
                )
            tensor = np.zeros((n,) * (power + 1))
            tensor[(np.arange(n),) * (power + 1)] = coefficients
            tensor.setflags(write=False)
            vertices.append(LocalVertex(f"F{power}", tensor))
        initial = None
        if self.start == "stationary":
            if not self._exchange:
                q0 = (
                    self.noise / (2 * self.gamma)
                    if self.gamma <= np.finfo(float).max / 2
                    else (0.5 * self.noise) / self.gamma
                )
            elif self._weak_forcing or (
                self._modes is None and self._exchange > 1000 * self.gamma
            ):
                q0 = np.asarray(
                    _coordinate_flow(self, 0.0, stationary=True)[1], dtype=float
                )
            elif self._modes is None:
                q0 = solve_continuous_lyapunov(self._drift, -self.noise)
            else:
                rates, vec, q = self._modes
                q0 = vec @ (q / (rates + rates[:, None])) @ vec.T
            initial = InitialState(covariance=0.5 * (q0 + q0.T))
        return System(
            FieldSpec("phi", n_components=n),
            ExplicitR(self._response, iso_R=False, vectorized=True),
            GaussianNoise(
                SeparableTranslation(ExponentialTemporal(0, 1), ExponentialSpatial(1)),
                sigma2=ConstantImpulse(self.noise),
            ),
            vertices=tuple(vertices),
            initial_state=initial,
            t_min=self.t_min,
        )

    def propagators(self, t_max):
        """Full causal response and two-time covariance, evaluated directly."""
        system = self.system
        covariance = self._driven
        if self._exchange and np.any(self.noise < 0):
            covariance = self._signed_covariance
            if self.start == "stationary":
                # This callable includes the stationary law in the same
                # higher-precision contraction as the driven response.
                system = replace(system, initial_state=None)
        return system.propagators(
            t_max=t_max,
            diag_C=False,
            c_closed_form=covariance,
            c_closed_form_only=True,
            c_closed_form_vectorized=True,
        )

    def expand(self, observable, orders, *, routing="multiplicity", target="moment"):
        """Expand using the node shared by the legs of each local vertex."""
        expansion = self.system.expand(
            observable,
            orders=orders,
            routing=routing,
            target=target,
            diag_R=False,
            diag_C=False,
        )
        diagrams = {
            order: [_site_indices(term) for term in terms]
            for order, terms in expansion.dts_by_order.items()
        }
        return replace(expansion, dts_by_order=diagrams)
