"""Spectral (disorder-averaged) propagators.

Motivation
----------
For a linear problem whose relaxation matrix has spectrum ``rho(h)``, the
*disorder-averaged* propagators are superpositions of Ornstein-Uhlenbeck ones:

.. math::

    R^*(t,t') &= \\Theta(t-t') \\int \\rho(h)\\, e^{-(h+\\lambda)(t-t')}\\,dh \\\\
    C^*(t_1,t_2) &= \\int \\rho(h)\\, \\frac{D}{h+\\lambda}
        \\left[ e^{-(h+\\lambda)|t_1-t_2|}
                - e^{-(h+\\lambda)(t_1+t_2-2t_{\\min})} \\right] dh

with the initial condition :math:`x(t_{\\min}) = 0` -- ``t_min`` is this
package's only initial-condition control, and hard-coding it at 0 while
accepting the argument was wrong by 7.4% at ``t_min = 1``.

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

Validity: this is an ANNEALED substitution above order 0
--------------------------------------------------------
Substituting :math:`\\langle R\\rangle` and :math:`\\langle C\\rangle` into an
interacting diagram is *not* a controlled quenched average.  It is exactly the
factorisation this module's own construction says fails --
:math:`\\langle R \\kappa R\\rangle \\neq
\\langle R\\rangle \\kappa \\langle R\\rangle` -- applied one level up, at the
vertex instead of at the propagator.  It is the annealed / one-loop-with-
dressed-lines step.

At order 0 there is nothing to average over and the result is exact.  Above
it, ``applications/ML/phase3`` measured the gap against the exact quenched
answer at **35%** -- not a small correction.  A controlled treatment needs
replicas or an explicit fluctuation expansion around the saddle, neither of
which this module provides.  Use it for order 0, for structure, and for cost
estimates; do not read an interacting order as a quenched result.

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


class _UniformKappa:
    """``kappa2`` for a cache that supplies ``C`` directly and is spatially
    uniform.

    Returns the IDENTITY, not zeros.  ``integrate_two_point_qmc`` forms a
    per-propagator spatial factor ``diag k2(n_l,n_r) / diag k2(0,0)``; with a
    zero ``kappa2`` the denominator trips that function's own
    ``|diag| > 1e-30`` guard and the factor collapses to 0, so **every
    two-point function at a nonzero separation silently returned exactly
    0.0** (measured: 0.4988 at r=0, 0.0 at r=1 and r=2.5).  The identity makes
    the ratio exactly 1, which is the honest statement here -- the
    disorder-averaged single-site theory this module represents has no spatial
    structure, so its ``C`` is the same at every separation.  See
    :class:`SpectralPropagatorCache` for what that means for the caller.

    A module-level class rather than a lambda so the cache stays picklable --
    it has to survive ``joblib.dump`` (``propagators.cache_path``) and loky.
    """

    __slots__ = ("n",)

    def __init__(self, n: int):
        self.n = int(n)

    def __call__(self, n1, t1, n2, t2):
        return np.eye(self.n)


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
        # Reject complex BEFORE casting: `np.asarray(z, dtype=float)` drops the
        # imaginary part with only a ComplexWarning, so a complex spectrum
        # would be silently truncated to its real part.
        for label, raw in (("nodes", self.nodes), ("weights", self.weights)):
            if np.iscomplexobj(np.asarray(raw)):
                raise ValueError(
                    f"{label} must be real; got a complex array.  A complex "
                    f"spectrum is not a relaxation-rate density -- take "
                    f".real explicitly if that is what you mean."
                )
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
        # `np.linalg.eigvalsh` on a rank-deficient Gram matrix returns
        # round-off negatives around -1e-16 -- and a sample covariance
        # spectrum is this module's advertised primary input, so a zero-
        # tolerance rejection would reject the main use case.  Clamp what is
        # numerically zero; reject what is genuinely negative.
        scale = float(np.max(np.abs(nodes))) if nodes.size else 1.0
        tol = 1e-10 * max(scale, 1.0)
        if np.any(nodes < -tol):
            raise ValueError(
                f"spectral nodes are relaxation RATES and must be >= 0; got a "
                f"minimum of {float(nodes.min()):.6g} against a tolerance of "
                f"{-tol:.3g}.  A negative rate is an unstable mode, for which "
                f"the stationary C does not exist."
            )
        nodes = np.clip(nodes, 0.0, None)
        total = float(weights.sum())
        if total <= 0:
            raise ValueError("weights must not sum to zero.")
        # Normalise -0.0 to 0.0.  `np.array_equal` treats them as equal but
        # `tobytes()` does not, so without this two densities could compare
        # equal and hash differently -- which breaks every dict and set.
        nodes = nodes + 0.0
        weights = weights / total + 0.0
        object.__setattr__(self, "nodes", np.where(nodes == 0.0, 0.0, nodes))
        object.__setattr__(
            self, "weights", np.where(weights == 0.0, 0.0, weights))

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

        The observed convergence rate is ~2 for a smooth density at a moderate
        time separation, but it is NOT a guarantee: the rate depends on how
        curved the integrand is across a bin, and adversarial review measured
        it dropping to ~0.9 on the same spectrum at a different time pair.
        Check convergence for your own density and time range rather than
        assuming a rate.

        Carrying every sampled eigenvalue instead would make each propagator
        call a reduction over the whole sample (200k in
        ``applications/ML/phase3``), inside every quadrature node of every
        diagram.
        """
        s = np.asarray(samples, dtype=float).ravel() + float(shift)
        if s.size == 0:
            raise ValueError("no samples given.")
        if int(n_nodes) < 1:
            raise ValueError(
                f"n_nodes must be >= 1; got {n_nodes!r}.  It used to collapse "
                f"silently to a single node, turning the whole spectrum into "
                f"its mean."
            )
        s = np.sort(s)
        n_nodes = int(min(n_nodes, s.size))
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

    def average(self, f: Callable[[np.ndarray], np.ndarray],
                node_axis: int = -1) -> np.ndarray:
        """``sum_i w_i f(h_i)``, contracting over the node axis.

        ``f`` is called once with the whole node array and must return
        something whose **last** axis is the node axis -- shape ``(n_nodes,)``
        for a scalar-valued ``f``, or ``(..., n_nodes)`` for a vector-valued
        one.  The natural per-node layout ``(n_nodes, k)`` is the transpose of
        that; it is caught only because its last axis has the wrong LENGTH,
        and numpy's own ``tensordot`` already raised on it, so the check buys a
        message rather than safety.

        When ``k == n_nodes`` the two layouts are indistinguishable by shape,
        and no check can tell them apart -- so say which axis you mean with
        ``node_axis`` instead of relying on the default.  Note that in exactly
        that square case the length check above is vacuous, so a wrong
        ``node_axis`` returns a wrong NUMBER rather than raising: the
        parameter is a declaration by the caller, not something the array can
        confirm.
        """
        vals = np.asarray(f(self.nodes), dtype=float)
        # Validate `node_axis` itself before indexing with it: an out-of-range
        # value would surface as a bare `IndexError: tuple index out of range`
        # from the shape lookup, and `node_axis=True` would sail past the
        # length check (`shape[True]` is `shape[1]`) only to die inside
        # tensordot.  Neither names the parameter at fault.
        if isinstance(node_axis, bool) or not isinstance(node_axis, (int,
                                                                     np.integer)):
            raise TypeError(
                f"node_axis must be an integer; got {node_axis!r}."
            )
        node_axis = int(node_axis)
        if vals.ndim == 0 or not (-vals.ndim <= node_axis < vals.ndim):
            raise ValueError(
                f"node_axis={node_axis} is out of range for the shape "
                f"{vals.shape} that f returned."
            )
        if vals.shape[node_axis] != self.n_nodes:
            raise ValueError(
                f"f must return an array whose axis {node_axis} is the node "
                f"axis (length {self.n_nodes}); got shape {vals.shape}.  Pass "
                f"`node_axis=` if the node axis is not the last one."
            )
        return np.tensordot(vals, self.weights, axes=([node_axis], [0]))

    def __eq__(self, other) -> bool:
        """Value equality.

        The generated dataclass ``__eq__`` compares ndarrays with ``==`` and
        then calls ``bool()`` on the result, which raises -- so a density could
        not be compared at all.
        """
        if not isinstance(other, SpectralDensity):
            return NotImplemented
        return (np.array_equal(self.nodes, other.nodes)
                and np.array_equal(self.weights, other.weights))

    def __hash__(self) -> int:
        """Hash on contents, so a density can key a memo or live in a set."""
        return hash((self.nodes.tobytes(), self.weights.tobytes(),
                     self.nodes.shape))

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
        if not np.isfinite(noise_D) or noise_D < 0:
            raise ValueError(
                f"noise_D is the noise amplitude D in <xi xi> = 2 D delta and "
                f"must be finite and >= 0; got {noise_D!r}."
            )
        if int(n_components) < 1:
            raise ValueError(
                f"n_components must be >= 1; got {n_components!r}."
            )
        if not np.isfinite(t_min):
            raise ValueError(f"t_min must be finite; got {t_min!r}.")
        self.density = density
        self.noise_D = float(noise_D)
        self._n = int(n_components)
        self._t_min = float(t_min)
        model = PropagatorModel(
            R_time=self._r_scalar,
            kappa2=_UniformKappa(self._n),
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

    def clear_cache(self) -> None:
        """Drop the memo but keep the batch-C sentinel.

        The inherited ``clear_cache`` sets ``_c_splines = None``, which for a
        table-backed cache means "the table is gone".  Here it is a capability
        flag -- ``C_diagonal_batch`` is a method, not a table -- so clearing it
        would silently demote every backend to the scalar Python loop.
        """
        super().clear_cache()
        self._c_splines = True

    def C_diagonal_batch(self, t1, t2) -> np.ndarray:
        """``(..., N)`` -- the component axis is APPENDED.

        ``np.atleast_1d(vals)[:, None]`` inserted it at position 1 instead,
        which is the same thing for the 1-D time arrays every backend passes
        but silently wrong for a 2-D one.
        """
        vals = np.atleast_1d(
            _c_star(self.density, t1, t2, self.noise_D, self._t_min))
        return np.repeat(vals[..., None], self._n, axis=-1)


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
