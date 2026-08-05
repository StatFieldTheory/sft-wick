"""Spectral (disorder-averaged) propagators.

Motivation
----------
For a linear problem whose relaxation matrix has spectrum ``rho(h)``, the
*disorder-averaged* propagators are superpositions of Ornstein-Uhlenbeck ones:

.. math::

    R^*(t,t') &= \\Theta(t-t') \\int \\rho(h)\\, e^{-(h+\\lambda)(t-t')}\\,dh \\\\
    C^*(t_1,t_2) &= \\int \\rho(h)\\, \\frac{D}{h+\\lambda}
        \\left[ e^{-(h+\\lambda)|t_1-t_2|} - e^{-(h+\\lambda)(t_1+t_2)} \\right] dh

These are genuinely non-exponential: :math:`R^*` is a superposition of decays,
so the effective single-site process is non-Markovian.  That is the structural
feature a DMFT solution has and a closed-form free theory does not.

Why this belongs in the library
-------------------------------
Two reasons, both learned the hard way in ``applications/ML/phase3``:

* ``C^*`` cannot be rebuilt from ``R^*`` through the package's own
  ``C = ∫∫ R κ R`` relation, because the ensemble average does not factorise:
  :math:`\\langle R \\kappa R\\rangle \\neq \\langle R\\rangle \\kappa \\langle R\\rangle`.
  So ``C`` must be injected independently -- which is what
  :class:`~sft_wick.evaluate.PropagatorCache`'s ``c_value_fn`` hook is for.
* Hand-rolling it invites two specific mistakes.  Tabulating ``C^*`` on a
  ``(t1,t2)`` grid and splining it reintroduces the **diagonal ridge**:
  ``C`` has a derivative discontinuity of exactly ``-2D`` on ``t1 == t2``
  (from the ``|t_1-t_2|``), which a tensor-product spline cannot represent, and
  every tadpole evaluates ``C(s,s)`` exactly there.  And Θ gets re-implemented
  by hand next to the library's own convention.  Evaluating the spectral sum
  directly avoids both: it is exact, and Θ comes from one place.

Cost
----
The point of the disorder-averaged route is that ``N`` disappears: the
effective problem is *scalar*, so the ``O(N^rank)`` coupling-index contraction
that dominates a per-instance matrix calculation is gone entirely.  What
replaces it is a sum over spectral nodes, which is why the density is reduced
to a small quadrature rule once (see :meth:`SpectralDensity.from_samples`)
rather than carrying every sampled eigenvalue into every propagator call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .evaluate import PropagatorCache, PropagatorModel

#: Below this rate the ``D/h`` prefactor is replaced by its finite ``h -> 0``
#: limit.  A zero mode is physical (free diffusion, ``C = 2 D (m - t_min)``);
#: what is not physical is a NEGATIVE rate, which :class:`SpectralDensity`
#: rejects.
_H_FLOOR = 1e-12

__all__ = ["SpectralDensity", "SpectralPropagatorCache", "spectral_cache"]


class _ZeroKappa:
    """``kappa2`` placeholder for a cache that supplies ``C`` directly.

    A module-level class rather than a lambda so the cache stays picklable --
    it has to survive ``joblib.dump`` (``propagators.cache_path``) and loky.
    """

    __slots__ = ("n",)

    def __init__(self, n: int):
        self.n = int(n)

    def __call__(self, n1, t1, n2, t2):
        return np.zeros((self.n, self.n))


@dataclass(frozen=True)
class SpectralDensity:
    """A spectral density reduced to nodes and weights.

    ``weights`` are normalised to sum to 1, so every spectral average is
    ``sum_i w_i f(h_i)`` -- an estimator of ``\\int rho(h) f(h) dh``.

    Construct with :meth:`from_samples` (empirical spectra, e.g. sampled
    Marchenko-Pastur eigenvalues), :meth:`from_callable` (an analytic density),
    or :meth:`delta` (a single relaxation rate, which reduces every formula
    here to the plain Ornstein-Uhlenbeck one).
    """

    nodes: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        nodes = np.asarray(self.nodes, dtype=float).ravel()
        weights = np.asarray(self.weights, dtype=float).ravel()
        if nodes.shape != weights.shape:
            raise ValueError(
                f"nodes and weights must have the same shape; got "
                f"{nodes.shape} and {weights.shape}."
            )
        if nodes.size == 0:
            raise ValueError("a spectral density needs at least one node.")
        if np.any(~np.isfinite(nodes)) or np.any(~np.isfinite(weights)):
            raise ValueError("nodes and weights must all be finite.")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative.")
        if np.any(nodes < 0):
            raise ValueError(
                f"spectral nodes are relaxation RATES and must be >= 0; got a "
                f"minimum of {float(nodes.min()):.6g}.  A negative rate is an "
                f"unstable mode, for which the stationary C does not exist."
            )
        total = float(weights.sum())
        if total <= 0:
            raise ValueError("weights must not sum to zero.")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "weights", weights / total)

    # -- constructors ------------------------------------------------- #

    @classmethod
    def delta(cls, h: float) -> "SpectralDensity":
        """A single relaxation rate — the Ornstein-Uhlenbeck limit."""
        return cls(np.array([float(h)]), np.array([1.0]))

    @classmethod
    def from_samples(
        cls, samples, n_nodes: int = 128, *, shift: float = 0.0,
    ) -> "SpectralDensity":
        """Reduce an empirical spectrum to ``n_nodes`` quadrature points.

        Uses equal-mass (quantile) binning with each node placed at its bin's
        mean, so the reduction is exact for any function that is linear across
        a bin and needs no assumption about the density's shape -- which
        matters for Marchenko-Pastur, whose edges are sharp.

        Carrying every sampled eigenvalue instead would make each propagator
        call a reduction over the whole sample (200k in
        ``applications/ML/phase3``), inside every quadrature node of every
        diagram.
        """
        s = np.asarray(samples, dtype=float).ravel() + float(shift)
        if s.size == 0:
            raise ValueError("no samples given.")
        s = np.sort(s)
        n_nodes = int(min(max(1, n_nodes), s.size))
        # Equal-mass bins: split the sorted samples into n_nodes contiguous
        # groups of (nearly) equal count.
        edges = np.linspace(0, s.size, n_nodes + 1).astype(int)
        nodes, weights = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            if b <= a:
                continue
            nodes.append(s[a:b].mean())
            weights.append(b - a)
        return cls(np.array(nodes), np.array(weights, dtype=float))

    @classmethod
    def from_callable(
        cls, rho: Callable[[Any], Any], lo: float, hi: float,
        n_nodes: int = 128, *, shift: float = 0.0,
    ) -> "SpectralDensity":
        """Gauss-Legendre reduction of an analytic density on ``[lo, hi]``.

        ``rho`` need not be normalised; the weights are renormalised anyway.
        """
        if not hi > lo:
            raise ValueError(f"need hi > lo; got lo={lo}, hi={hi}.")
        x, w = np.polynomial.legendre.leggauss(int(n_nodes))
        half = 0.5 * (hi - lo)
        h = lo + half * (x + 1.0)
        dens = np.asarray(rho(h), dtype=float)
        if dens.shape != h.shape:
            raise ValueError(
                "rho must be vectorised: rho(array) should return an array of "
                f"the same shape; got {dens.shape} for input {h.shape}."
            )
        if np.any(dens < 0):
            raise ValueError("rho returned a negative value.")
        return cls(h + float(shift), w * half * dens)

    # -- spectral averages -------------------------------------------- #

    def average(self, f: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """``sum_i w_i f(h_i)`` with ``f`` broadcast over the nodes."""
        vals = np.asarray(f(self.nodes), dtype=float)
        return np.tensordot(vals, self.weights, axes=([-1], [0]))

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.size)

    def __len__(self) -> int:
        return self.n_nodes


def _r_star(density: SpectralDensity, dt) -> np.ndarray:
    """``<exp(-h dt)>_rho`` for ``dt >= 0``; Θ is applied by the caller."""
    dt_arr = np.asarray(dt, dtype=float)
    # NOT clamped at 0: `PropagatorCache.R_time` is the raw, Theta-stripped
    # accessor by this package's convention, and clamping would silently make
    # it return 1 for an acausal argument instead of the analytic
    # continuation.  Theta is applied by the batch accessor and by the three
    # diagram-side consumers, none of which call this for an acausal pair.
    # (..., 1) x (n_nodes,) -> contract the node axis.
    return np.tensordot(
        np.exp(-dt_arr[..., None] * density.nodes),
        density.weights, axes=([-1], [0]),
    )


def _c_star(density: SpectralDensity, t1, t2, noise_D: float,
            t_min: float = 0.0) -> np.ndarray:
    """``<(D/h)(exp(-h|t1-t2|) - exp(-h(t1+t2-2 t_min)))>_rho``.

    Derived from the definition rather than assumed::

        C(t1,t2) = int_{t_min}^{m} R(t1,l) 2D R(t2,l) dl,   m = min(t1,t2)
                 = 2D e^{-h(t1+t2)} (e^{2hm} - e^{2h t_min}) / (2h)
                 = (D/h) (e^{-h|t1-t2|} - e^{-h(t1+t2-2 t_min)})

    using ``t1 + t2 - 2 min(t1,t2) = |t1 - t2|``.  The ``t_min`` term is the
    initial condition ``x(t_min) = 0``; hard-coding it at 0 while accepting a
    ``t_min`` argument was wrong by 1.5% at ``t_min = 0.5`` and 7.4% at 1.0.

    Evaluated exactly, never tabulated: the ``|t1-t2|`` gives ``C`` a
    derivative discontinuity of ``-2D`` on the diagonal, and splining across
    that ridge is what stops a tabulated ``C`` from converging where every
    tadpole evaluates it.

    Two numerical points:

    * the two exponentials nearly cancel for small ``h``, so the difference
      goes through ``expm1``;
    * ``D/h`` diverges as ``h -> 0``, but ``C`` does not -- the limit is
      ``2 D (m - t_min)``, free diffusion.  Nodes below ``_H_FLOOR`` take it.
    """
    a = np.asarray(t1, dtype=float)
    b = np.asarray(t2, dtype=float)
    h = density.nodes

    # delta = 2 (min(t1,t2) - t_min), clamped: before t_min no noise has been
    # integrated, so C is 0 there rather than negative.
    delta = np.clip(2.0 * (np.minimum(a, b) - float(t_min)), 0.0, None)
    x = np.abs(a - b)[..., None] * h                     # h |t1 - t2|
    z = delta[..., None] * h                             # h * delta

    small = np.abs(h) < _H_FLOOR
    h_safe = np.where(small, 1.0, h)
    # (D/h) (e^{-x} - e^{-(x+z)}) = D e^{-x} (-expm1(-z)) / h
    kernel = noise_D * np.exp(-x) * np.where(
        small, delta[..., None], -np.expm1(-z) / h_safe,
    )
    return np.tensordot(kernel, density.weights, axes=([-1], [0]))


class SpectralPropagatorCache(PropagatorCache):
    """A cache whose ``R`` and ``C`` are spectral superpositions.

    Both are evaluated exactly from the density -- there is no interpolation
    table, so there is no diagonal ridge to resolve and nothing to keep in
    sync.  The batch accessors are true vectorised reductions rather than
    ``np.vectorize`` over a scalar callable.

    ``_c_splines`` is set to a sentinel because the batch integrators use it to
    mean "batch C is available"; the scalar accessors are overridden alongside
    so nothing tries to read a table range that does not exist.
    """

    def __init__(
        self, density: SpectralDensity, noise_D: float, *,
        n_components: int = 1, t_min: float = 0.0,
    ):
        self.density = density
        self.noise_D = float(noise_D)
        self._n = int(n_components)
        self._t_min = float(t_min)
        model = PropagatorModel(
            R_time=self._r_scalar,
            kappa2=_ZeroKappa(self._n),
            n_components=self._n, iso_R=True, diag_C=True, t_min=float(t_min),
        )
        super().__init__(model, c_value_fn=self._c_matrix)
        self._c_splines = True  # sentinel: batch C is available

    # -- R --------------------------------------------------------------- #

    def _r_scalar(self, t: float, tp: float) -> float:
        """Raw, Θ-stripped, per this package's convention (see
        ``PropagatorCache.R_time``): Θ is applied at diagram evaluation."""
        return float(_r_star(self.density, float(t) - float(tp)))

    def R_time_batch(self, t1, t2) -> np.ndarray:
        t1a = np.asarray(t1, dtype=float)
        t2a = np.asarray(t2, dtype=float)
        out = np.zeros(np.broadcast(t1a, t2a).shape, dtype=float)
        causal = t1a > t2a
        if np.any(causal):
            b1, b2 = np.broadcast_arrays(t1a, t2a)
            out[causal] = _r_star(self.density, b1[causal] - b2[causal])
        return out

    # -- C --------------------------------------------------------------- #

    def _c_matrix(self, n1, t1, n2, t2) -> np.ndarray:
        return np.eye(self._n) * float(
            _c_star(self.density, float(t1), float(t2), self.noise_D,
                    self._t_min)
        )

    def C_value(self, n1, t1, n2, t2) -> np.ndarray:
        return self._c_matrix(n1, t1, n2, t2)

    def C_diagonal(self, n, t1, n_prime=None, t2=None) -> np.ndarray:
        tb = t1 if t2 is None else t2
        return np.full(
            self._n,
            float(_c_star(self.density, float(t1), float(tb), self.noise_D,
                          self._t_min)),
        )

    def C_diagonal_batch(self, t1, t2) -> np.ndarray:
        vals = _c_star(self.density, t1, t2, self.noise_D, self._t_min)
        return np.repeat(np.atleast_1d(vals)[:, None], self._n, axis=1)


def spectral_cache(
    density: SpectralDensity | Any, noise_D: float, *,
    shift: float = 0.0, n_components: int = 1, t_min: float = 0.0,
) -> SpectralPropagatorCache:
    """Build a :class:`SpectralPropagatorCache`.

    ``density`` may be a :class:`SpectralDensity`, a single float (the
    Ornstein-Uhlenbeck limit), or an array of sampled eigenvalues.  ``shift``
    is added to every rate, for the ``h + lambda`` that a regulariser or weight
    decay contributes.
    """
    if not isinstance(density, SpectralDensity):
        arr = np.asarray(density, dtype=float).ravel()
        density = (SpectralDensity.delta(float(arr[0]) + shift)
                   if arr.size == 1
                   else SpectralDensity.from_samples(arr, shift=shift))
    elif shift:
        density = SpectralDensity(density.nodes + float(shift),
                                  density.weights)
    return SpectralPropagatorCache(
        density, noise_D, n_components=n_components, t_min=t_min,
    )
