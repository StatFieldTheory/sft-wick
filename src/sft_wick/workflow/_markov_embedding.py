"""Lower finite temporal kernels and memory to a constant matrix SDE."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, replace
from numbers import Integral, Real
from typing import Literal

import numpy as np

from .._contraction_admission import DEFAULT_CONTRACTION_MAX_BYTES
from ._covariance import is_covariance
from .specs import (
    ConstantImpulse,
    DampedCosineTemporal,
    DiagonalA,
    ExplicitR,
    ExponentialTemporal,
    LocalVertex,
    Matern32Temporal,
    SeparableTranslation,
)


def _finite_real(value: Real, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite real number.")
    return number


@dataclass(frozen=True)
class PronyMemory:
    """Memory kernel Σ_j weights[j] exp(−rates[j] t) for t >= 0.

    Each auxiliary variable obeys dz_j = (weights[j] x − rates[j] z_j) dt.
    A positive temperature adds independent white noise to each auxiliary
    variable, with covariance rate 2 temperature rates[j] weights[j].
    The physical drift receives the sum of the auxiliary variables times
    the signed coupling of the process.
    """

    rates: tuple[float, ...]
    weights: tuple[float, ...]
    temperature: float = 0.0

    def __post_init__(self) -> None:
        rates = tuple(_finite_real(item, "PronyMemory rate") for item in self.rates)
        weights = tuple(_finite_real(item, "PronyMemory weight") for item in self.weights)
        temperature = _finite_real(self.temperature, "PronyMemory temperature")
        if not rates or len(rates) != len(weights):
            raise ValueError("PronyMemory rates and weights must have the same length ≥ 1.")
        if min(rates) <= 0.0:
            raise ValueError("PronyMemory rates must be positive.")
        if temperature < 0.0 or (temperature > 0.0 and min(weights) < 0.0):
            raise ValueError(
                "PronyMemory temperature and weights at positive temperature must be non-negative."
            )
        object.__setattr__(self, "rates", rates)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "temperature", temperature)


@dataclass(frozen=True)
class ErlangDelay:
    """Unit-mass Erlang delay density with mean mean_delay."""

    mean_delay: float
    stages: int = 3

    def __post_init__(self) -> None:
        delay = _finite_real(self.mean_delay, "ErlangDelay mean_delay")
        if delay <= 0.0:
            raise ValueError("ErlangDelay mean_delay must be positive.")
        if (isinstance(self.stages, (bool, np.bool_))
                or not isinstance(self.stages, Integral) or self.stages < 1):
            raise ValueError("ErlangDelay stages must be a positive integer.")
        if not math.isfinite(self.stages / delay):
            raise ValueError("ErlangDelay stages / mean_delay must be finite.")
        object.__setattr__(self, "mean_delay", delay)
        object.__setattr__(self, "stages", int(self.stages))


_NOISE_KERNELS = (ExponentialTemporal, Matern32Temporal, DampedCosineTemporal)
_MEMORY_KERNELS = (PronyMemory, ErlangDelay)
_MAX_COMPONENTS = 64
EmbeddingKernel = (
    ExponentialTemporal | Matern32Temporal | DampedCosineTemporal | PronyMemory | ErlangDelay
)


@dataclass(frozen=True)
class EmbeddedProcess:
    """A temporal kernel acting on component target of the physical field.

    For a noise kernel, coupling loads the kernel's scalar output ξ into
    the target drift. In random_rate mode the term is coupling * x_target
    * ξ and carries the perturbative tag name. A square matrix coupling G
    gives ξ G x across all physical components with one shared noise.
    For a memory kernel the
    output is the convolution of the source component with the kernel,
    including its initial memory. Source defaults to target.

    A zero start initializes all hidden components to zero. A stationary
    start uses the stationary hidden law for a noise kernel, or the joint
    Gaussian stationary law of the full stable system for a memory kernel.
    """

    kernel: EmbeddingKernel
    target: int = 0
    coupling: float | tuple[tuple[float, ...], ...] = 1.0
    mode: Literal["additive", "random_rate"] = "additive"
    start: Literal["zero", "stationary"] = "zero"
    source: int | None = None
    name: str = "embedded"

    def __post_init__(self) -> None:
        if not isinstance(self.kernel, _NOISE_KERNELS + _MEMORY_KERNELS):
            raise ValueError("EmbeddedProcess needs a supported finite Markov kernel.")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("EmbeddedProcess name must be a nonempty string.")
        if self.mode not in ("additive", "random_rate"):
            raise ValueError("EmbeddedProcess mode must be 'additive' or 'random_rate'.")
        if self.start not in ("zero", "stationary"):
            raise ValueError("EmbeddedProcess start must be 'zero' or 'stationary'.")
        if isinstance(self.kernel, _MEMORY_KERNELS) and self.mode != "additive":
            raise ValueError("Memory kernels do not support random rate modes.")
        if isinstance(self.kernel, _NOISE_KERNELS):
            if self.source is not None:
                raise ValueError("A noise kernel cannot take a physical source component.")
            if _finite_real(self.kernel.lam, "EmbeddedProcess variance") < 0.0:
                raise ValueError("EmbeddedProcess variance must be non-negative.")
        if isinstance(self.coupling, (list, tuple, np.ndarray)):
            matrix = self.coupling
            if self.mode != "random_rate" or self.target != 0:
                raise ValueError("A matrix coupling requires random_rate and the default target.")
            if isinstance(matrix, np.ndarray) and matrix.ndim != 2:
                raise ValueError("The matrix coupling must have two axes.")
            n = len(matrix)
            if not 0 < n < _MAX_COMPONENTS:
                raise ValueError("The matrix coupling must fit the physical components.")
            rows = []
            for row in matrix:
                if (not isinstance(row, (list, tuple, np.ndarray))
                        or (isinstance(row, np.ndarray) and row.ndim != 1)
                        or len(row) != n):
                    raise ValueError("The matrix coupling must have one entry per physical component in each row.")
                rows.append(tuple(_finite_real(v, "matrix coupling") for v in row))
            coupling = tuple(rows)
        else:
            coupling = _finite_real(self.coupling, "EmbeddedProcess coupling")
        object.__setattr__(self, "coupling", coupling)


def _noise_realization(kernel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hidden drift, covariance rate and stationary covariance."""
    variance = float(kernel.lam)
    if isinstance(kernel, Matern32Temporal):
        # The companion state is ξ′/rate. This scaling keeps P = λI.
        rate = math.sqrt(3.0) / float(kernel.sigma_t)
        return (
            np.array([[0.0, rate], [-rate, -2.0 * rate]]),
            np.diag([0.0, 4.0 * (rate * variance)]),
            variance * np.eye(2),
        )
    rate = 1.0 / float(kernel.sigma_t)
    if isinstance(kernel, DampedCosineTemporal):
        omega = float(kernel.omega)
        return (
            np.array([[-rate, -omega], [omega, -rate]]),
            2.0 * (rate * variance) * np.eye(2),
            variance * np.eye(2),
        )
    return np.array([[-rate]]), np.array([[2.0 * (rate * variance)]]), np.array([[variance]])


def _memory_realization(kernel) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Hidden drift, covariance rate, linear input and linear output."""
    if isinstance(kernel, PronyMemory):
        rates, weights = np.asarray(kernel.rates), np.asarray(kernel.weights)
        noise = 2.0 * kernel.temperature * weights * rates
        return -np.diag(rates), np.diag(noise), weights, np.ones_like(weights)
    rate = kernel.stages / kernel.mean_delay
    n = kernel.stages
    drift = rate * (np.eye(n, k=-1) - np.eye(n))
    inp, out = np.eye(n)[[0, n - 1]]
    return drift, np.zeros((n, n)), rate * inp, out


def _transpose_last(x: np.ndarray) -> np.ndarray:
    return np.swapaxes(x, -1, -2)


@dataclass(frozen=True, eq=False)
class _DrivenCovariance:
    drift: np.ndarray
    diffusion: np.ndarray
    t_min: float

    @property
    def has_diagonal_kink(self) -> bool:
        return bool(np.any(self.diffusion))

    def __call__(self, x, t, y, s) -> np.ndarray:
        from scipy.linalg import expm

        # The augmented driving is shared across the spatial positions.
        t, s = np.broadcast_arrays(t, s)
        early = np.minimum(t, s)
        elapsed = np.maximum(0.0, early - self.t_min)
        cov = self._same_time(elapsed)
        return expm((t - early)[..., None, None] * self.drift) @ cov @ expm(
            (s - early)[..., None, None] * self.drift.T
        )

    def _same_time(self, elapsed: np.ndarray) -> np.ndarray:
        """Integrate the driven covariance by Van Loan and time doubling."""
        from scipy.linalg import expm

        n = len(self.drift)
        # exp([[A, Q], [0, -Aᵀ]] * t) contains a backward (antidamped)
        # block. Keep that exponential on short steps, then double time
        # with the forward semigroup: C(2h) = C(h) + R(h) C(h) R(h)ᵀ.
        scale = np.max(elapsed, initial=0.0) * np.linalg.norm(self.drift, ord=1)
        if not np.isfinite(scale):
            raise ValueError("The time range overflows the Markov embedding's drift scale.")
        doublings = math.ceil(math.log2(scale)) if scale > 1.0 else 0
        step = np.ldexp(elapsed, -doublings)
        diffusion_scale = max(1.0, float(np.max(np.abs(self.diffusion))))
        diffusion = self.diffusion / diffusion_scale
        if np.any((diffusion == 0.0) & (self.diffusion != 0.0)):
            raise ValueError("The Markov embedding diffusion spans too many scales.")
        block = np.block([[self.drift, diffusion],
                          [np.zeros_like(self.drift), -self.drift.T]])
        value = expm(step[..., None, None] * block)
        cov = value[..., :n, n:] @ _transpose_last(value[..., :n, :n])
        for power in range(doublings, 0, -1):
            # Recompute R rather than square a factor rounded to one on
            # a slow mode of a very disparate system.
            forward = expm(np.ldexp(elapsed, -power)[..., None, None] * self.drift)
            cov = cov + forward @ cov @ _transpose_last(forward)
        cov = (0.5 * cov + 0.5 * _transpose_last(cov)) * diffusion_scale
        if not np.all(np.isfinite(cov)):
            raise ValueError("The Markov embedding driven covariance is not numerically finite.")
        return cov


@dataclass(frozen=True, eq=False)
class CompiledMarkovEmbedding:
    """Augmented coefficients for the existing R, C and local-vertex routes."""

    drift: np.ndarray
    diffusion: np.ndarray
    initial_covariance: np.ndarray
    physical_components: int
    vertex: LocalVertex | None

    def response(self, t, s) -> np.ndarray:
        """The matrix exponential before applying the engine's causality rule."""
        from scipy.linalg import expm

        lag = np.maximum(np.subtract(t, s), 0.0)
        return expm(lag[..., None, None] * self.drift)

    def driven_covariance(self, t_min: float) -> _DrivenCovariance:
        """Covariance from the white driving alone. InitialState adds its own C."""
        return _DrivenCovariance(self.drift, self.diffusion, _finite_real(t_min, "t_min"))


@dataclass(frozen=True)
class _EmbeddedR(ExplicitR):
    compiled: CompiledMarkovEmbedding = field(kw_only=True)


def embed_system(system, process: EmbeddedProcess):
    """Compile a finite kernel on a constant-rate, white-driven physical system.

    The physical components keep their indices. The hidden states follow
    them and are integrated by the existing diagram contraction.
    """
    from ._initial_state import InitialState

    linear = system.linear
    if not isinstance(linear, DiagonalA) or callable(linear.gamma) or system.explicit_R is not None:
        raise ValueError("Markov embedding requires constant physical diagonal rates.")
    n = system.n_components
    rates = np.broadcast_to(np.asarray(linear.gamma, dtype=float), (n,))
    kappa = system.noise.kappa2
    if (
        not isinstance(kappa, SeparableTranslation)
        or type(kappa.temporal) not in _NOISE_KERNELS
        or kappa.temporal.lam != 0.0
    ):
        raise ValueError("Markov embedding requires zero smooth driving in the physical system.")
    white = system.noise.sigma2
    if white is not None and type(white) is not ConstantImpulse:
        raise ValueError("Markov embedding requires constant physical white driving.")
    diffusion = np.zeros((n, n)) if white is None else np.asarray(white.amplitude)
    if diffusion.ndim == 0:
        diffusion = float(diffusion) * np.eye(n)
    compiled = compile_markov_embedding(-np.diag(rates), diffusion, process)
    hidden = len(compiled.drift) - n

    def pad(tensor):
        return np.pad(tensor, [(0, hidden)] * np.ndim(tensor))

    def pad_vertices(vertices, scalar_rank):
        return tuple(
            replace(v, coupling=pad(
                np.reshape(v.coupling, (1,) * scalar_rank(v))
                if np.ndim(v.coupling) == 0 else v.coupling
            ))
            for v in vertices
        )

    if any(
        callable(v.coupling)
        for v in (*system.vertices, *system.nonlocal_vertices, *system.km_vertices, *system.grouped_vertices)
    ):
        raise ValueError("Markov embedding requires static physical vertex tensors.")
    if any(v.already_R_contracted for v in system.nonlocal_vertices):
        raise ValueError("Markov embedding requires bare physical cumulant tensors.")
    ranks = [v.n_legs for v in (*system.vertices, *system.km_vertices, *system.grouped_vertices)]
    ranks.extend(v.order for v in system.nonlocal_vertices)
    if system.initial_state is not None:
        ranks.extend(tensor.ndim for tensor in system.initial_state.cumulants)
    size = n + hidden
    if sum(size**rank for rank in ranks) * np.dtype(float).itemsize > DEFAULT_CONTRACTION_MAX_BYTES:
        raise ValueError("Markov embedding would exceed the augmented tensor memory limit.")
    vertices = pad_vertices(system.vertices, lambda v: v.n_legs)
    if compiled.vertex is not None:
        vertices = (*vertices, compiled.vertex)
    initial = system.initial_state
    covariance = compiled.initial_covariance
    if initial is not None:
        if process.start == "stationary" and isinstance(process.kernel, _MEMORY_KERNELS):
            raise ValueError("Stationary memory already specifies the full initial law.")
        if initial.covariance is not None:
            covariance = covariance + pad(initial.covariance)
    initial = InitialState(
        covariance=covariance if np.any(covariance) else None,
        cumulants=() if initial is None else tuple(map(pad, initial.cumulants)),
        mean=None if initial is None or initial.mean is None else pad(initial.mean),
    )
    return replace(
        system,
        field=replace(system.field, n_components=n + hidden),
        linear=_EmbeddedR(compiled.response, iso_R=False, vectorized=True, compiled=compiled),
        noise=replace(system.noise, sigma2=ConstantImpulse(compiled.diffusion)),
        vertices=vertices,
        nonlocal_vertices=pad_vertices(system.nonlocal_vertices, lambda v: v.order),
        km_vertices=pad_vertices(system.km_vertices, lambda v: v.n_legs),
        grouped_vertices=pad_vertices(system.grouped_vertices, lambda v: v.n_legs),
        initial_state=initial,
    )


def _as_real_matrix(values, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a finite real matrix.")
    matrix = np.asarray(raw, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.size == 0:
        raise ValueError(f"{name} must be a non-empty square matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite real matrix.")
    return matrix


def _component(index: int, n: int, name: str) -> int:
    if (isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral)
            or index < 0 or index >= n):
        raise ValueError(f"EmbeddedProcess {name} must index a physical component (0..{n - 1}).")
    return int(index)


def _stationary_covariance(drift: np.ndarray, diffusion: np.ndarray) -> np.ndarray:
    from scipy.linalg import solve_continuous_lyapunov

    if np.max(np.linalg.eigvals(drift).real) >= 0.0:
        raise ValueError("Stationary Markov memory needs a stable full drift matrix.")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        covariance = solve_continuous_lyapunov(drift, -diffusion)
    covariance = 0.5 * covariance + 0.5 * covariance.T
    if caught or not is_covariance(covariance):
        # LAPACK's triangular Sylvester solve can perturb a small but
        # well-determined denominator relative to unrelated faster modes.
        identity = np.eye(len(drift))
        pair_drift = np.kron(drift, identity) + np.kron(identity, drift)
        try:
            covariance = np.linalg.solve(pair_drift, -diffusion.ravel()).reshape(drift.shape)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Could not solve the stationary Markov memory covariance.") from exc
        covariance = 0.5 * covariance + 0.5 * covariance.T
    residual = drift @ covariance + covariance @ drift.T + diffusion
    scale = (1e-10 * np.abs(drift)) @ np.abs(covariance)
    budget = scale + scale.T + 1e-10 * np.abs(diffusion)
    # Off-diagonal elements that vanish exactly can accumulate roundoff
    # from other states in the Schur solve. Allow their component scales.
    component_budget = np.sqrt(np.diag(budget))
    budget += np.multiply.outer(component_budget, component_budget)
    if (not is_covariance(covariance) or not np.all(np.isfinite(budget))
            or not np.all(np.abs(residual) <= budget)):
        raise ValueError("Could not verify the stationary Markov memory covariance.")
    return covariance


def compile_markov_embedding(drift, diffusion, process: EmbeddedProcess) -> CompiledMarkovEmbedding:
    """Append one process to an SDE with a constant drift and white diffusion.

    drift is the signed physical matrix in dx = A x dt + B dW.
    diffusion is the covariance rate BBᵀ of the physical drivers.
    Hidden indices follow the physical indices. Noise drivers of the
    physical and hidden states are independent, with no spatial dependence.
    """
    physical_drift = _as_real_matrix(drift, "Markov embedding drift")
    physical_diffusion = _as_real_matrix(diffusion, "Markov embedding diffusion")
    if physical_diffusion.shape != physical_drift.shape:
        raise ValueError("The Markov embedding drift and diffusion shapes must match.")
    if not np.allclose(physical_diffusion, physical_diffusion.T, rtol=1e-12, atol=0.0):
        raise ValueError("Markov embedding diffusion must be symmetric.")
    physical_diffusion = 0.5 * physical_diffusion + 0.5 * physical_diffusion.T
    if not is_covariance(physical_diffusion):
        raise ValueError("Markov embedding diffusion must be positive semidefinite.")
    n = len(physical_drift)
    target = _component(process.target, n, "target")
    kernel = process.kernel
    memory = isinstance(kernel, _MEMORY_KERNELS)
    hidden = (
        len(kernel.rates) if isinstance(kernel, PronyMemory)
        else kernel.stages if isinstance(kernel, ErlangDelay)
        else 1 if isinstance(kernel, ExponentialTemporal)
        else 2
    )
    if n + hidden > _MAX_COMPONENTS:
        raise ValueError(f"Markov embedding supports up to {_MAX_COMPONENTS} total components.")
    if memory:
        hidden_drift, hidden_diffusion, inp, out = _memory_realization(kernel)
    else:
        hidden_drift, hidden_diffusion, hidden_stationary = _noise_realization(kernel)

    size = n + len(hidden_drift)
    augmented_drift = np.zeros((size, size))
    augmented_diffusion = np.zeros_like(augmented_drift)
    covariance = np.zeros_like(augmented_drift)
    augmented_drift[:n, :n], augmented_drift[n:, n:] = physical_drift, hidden_drift
    augmented_diffusion[:n, :n], augmented_diffusion[n:, n:] = physical_diffusion, hidden_diffusion
    vertex = None
    if memory:
        source = target if process.source is None else _component(process.source, n, "source")
        augmented_drift[n:, source] = inp
        augmented_drift[target, n:] = process.coupling * out
    elif process.mode == "additive":
        augmented_drift[target, n] = process.coupling
    else:
        tensor = np.zeros((size, size, size))
        if isinstance(process.coupling, tuple):
            if len(process.coupling) != n:
                raise ValueError("The matrix coupling must match the physical component count.")
            half = 0.5 * np.asarray(process.coupling)
            tensor[:n, :n, n] = tensor[:n, n, :n] = half
        else:
            tensor[target, target, n] = tensor[target, n, target] = 0.5 * process.coupling
        tensor.setflags(write=False)
        vertex = LocalVertex(process.name, tensor)
    if not np.all(np.isfinite(augmented_drift)) or not np.all(np.isfinite(augmented_diffusion)):
        raise ValueError("Markov embedding kernel parameters overflow the drift or diffusion.")

    if process.start == "stationary":
        if memory:
            covariance = _stationary_covariance(augmented_drift, augmented_diffusion)
        else:
            covariance[n:, n:] = hidden_stationary
    for matrix in (augmented_drift, augmented_diffusion, covariance):
        matrix.setflags(write=False)
    return CompiledMarkovEmbedding(augmented_drift, augmented_diffusion, covariance, n, vertex)
