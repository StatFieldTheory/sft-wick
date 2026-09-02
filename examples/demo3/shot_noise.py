r"""Filtered-Poisson (shot) noise: exact cumulants and R-contracted vertices.

Demo 3's driving field is a **filtered Poisson process**.  Independently
for each component ``a``, events ``(x_k, s_k)`` are drawn from a Poisson
process of rate ``ν`` per unit length × time on the whole line and the
whole past, and

.. math::

    η_a(x, t) = Σ_k h · w(x − x_k) · g(t − s_k) − ⟨·⟩,
    \qquad g(τ) = Θ(τ) e^{−τ/σ_t}, \qquad w(x) = e^{−|x|/σ_x}.

Campbell's theorem gives **every** cumulant in closed form, all with the
same shape --- a spatial overlap times a temporal overlap, both of which
factor through the single source point ``(x', s)``::

    κ_m(z_1 … z_m) = δ_{a_1 … a_m} · ν h^m · X_m(x_1 … x_m) · T_m(t_1 … t_m)
    X_m = ∫dx' Π_i w(x_i − x')            (coincident: 2σ_x/m)
    T_m = ∫ds  Π_i g(t_i − s) = (σ_t/m) · exp(−Σ_i (t_i − t_min)/σ_t)

Two consequences make this model a much cleaner validation target than a
deformed-Gaussian field:

1. **One dimensionless knob.**  With ``n ≡ ν σ_t σ_x`` the shape of the
   one-point law depends on ``n`` alone::

       skewness = (4√2/9) / √n ≈ 0.6285/√n,   excess kurtosis = 0.5/n,

   and ``n → ∞`` is the Gaussian limit.  Nothing else moves: ``κ²`` is
   held fixed by compensating ``h``.

2. **The R-contracted vertex is closed form.**  Because every cumulant
   factors through the single source time ``s``, the ``m`` leg integrals
   commute with it and collapse to a *one*-dimensional integral::

       K_R(t'; x') = ∫ Π_i du_i R(t'_i, u_i) κ_m(u; x')
                   = ν h^m · X_m(x') · T̃_m(t'),
       T̃_m(t')     = ∫ ds Π_i J(t'_i, s),
       J(t', s)    = ∫_0^{t'} R(t', u) g(u − s) du.

   ``J`` is elementary and ``T̃_m`` has a closed form (:func:`t_tilde_closed`,
   ``2^m`` exponential terms).  Contrast demo 2, whose R-contracted κ³
   needs a cusp-aligned composite rule accurate only to ~1e-6.

Numerics
--------
``T̃_m``'s closed form carries an explicit ``(γ − a)^{−m}`` with
``a = 1/σ_t``: a *removable* singularity at ``γ = a``, but a genuine
cancellation in floating point, losing roughly ``m · log10(a/|γ − a|)``
digits.  :func:`t_tilde` therefore dispatches on
``rel = |γ − a| / max(γ, a)``:

===================  ====================================================
``rel ≥ RE L_SPLIT``  :func:`t_tilde_closed` --- exact, ``2^m`` terms
``rel <  REL_SPLIT``  :func:`t_tilde_quad` --- graded composite Gauss-
                      Legendre in ``s``, uniformly stable, ~25x slower
===================  ====================================================

Both branches are valid in a neighbourhood of the threshold and are
compared *directly* there by
``tests/test_demo3_shot_noise.py::test_boundary_*`` (the boundary-
validation methodology: never test a dispatcher through itself).  At the
demo's own parameters (``σ_t = 0.5``, ``γ = 1`` → ``rel = 0.5``) the two
agree to ≲ 5e-12 and both agree with a fully independent ``m``-dimensional
adaptive integral of the raw leg integral to ~1e-15.

All array-valued helpers take ``(m, n_samples)`` inputs and return
``(n_samples,)``; scalars are accepted via ``np.atleast_2d``.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss

__all__ = [
    "ShotNoise", "PARAMS", "N_COMP",
    "X_m", "T_m", "kappa_m", "J", "A_response", "t_tilde",
    "t_tilde_closed", "t_tilde_quad", "K_R",
    "kappa2_spatial", "kappa2_lam",
    "coupling_k3", "coupling_k3_vectorized",
    "coupling_k4", "coupling_k4_vectorized",
]

#: Relative-error budget for :func:`t_tilde`'s ``'auto'`` dispatch.  The
#: closed form *estimates its own conditioning* (see
#: :func:`t_tilde_closed`), and any sample whose estimate exceeds this is
#: recomputed by quadrature.  There is deliberately **no threshold in
#: ``|γ − a|`` alone**: the two branches fail in disjoint corners --- the
#: closed form when the result is tiny against its individual terms
#: (small ``t'``), the quadrature when its panel stack would truncate
#: early (large ``T``) --- so a constant in ``γ`` cannot separate them.
AUTO_TOL = 1e-12

#: Machine epsilon used in the conditioning estimate.
_EPS = float(np.finfo(float).eps)

#: Gauss-Legendre nodes per panel and decades of decay covered by the
#: graded panel stack in :func:`t_tilde_quad`.
N_GL_PANEL = 8
DECADES = 42.0


# =====================================================================
# Parameters
# =====================================================================

@dataclass(frozen=True)
class ShotNoise:
    """Parameters of the filtered-Poisson driving field and the drift.

    Args:
        nu: event rate ``ν`` per unit length per unit time.
        h: pulse amplitude ``h``.
        sigma_t: pulse decay time ``σ_t`` (``g(τ) = Θ(τ)e^{−τ/σ_t}``).
        sigma_x: pulse width ``σ_x`` (``w(x) = e^{−|x|/σ_x}``).
        gamma: linear drift rate ``γ`` (``R(t,u) = Θ(t−u)e^{−γ(t−u)}``).
        n_components: number of field components ``N``.
    """

    nu: float = 2.0
    h: float = 1.0
    sigma_t: float = 0.5
    sigma_x: float = 1.0
    gamma: float = 1.0
    n_components: int = 2

    # -- derived ------------------------------------------------------

    @property
    def a(self) -> float:
        """Pulse decay rate ``a = 1/σ_t``."""
        return 1.0 / self.sigma_t

    @property
    def n_dimensionless(self) -> float:
        """``n = ν σ_t σ_x`` --- the single non-Gaussianity knob."""
        return self.nu * self.sigma_t * self.sigma_x

    @property
    def variance(self) -> float:
        """``κ₂`` at coincident points, ``= n h² / 2``."""
        return self.nu * self.h ** 2 * self.sigma_x * self.sigma_t / 2.0

    @property
    def skewness(self) -> float:
        """``κ₃/κ₂^{3/2} = (4√2/9)/√n`` --- independent of ``h``."""
        return (4.0 * np.sqrt(2.0) / 9.0) / np.sqrt(self.n_dimensionless)

    @property
    def excess_kurtosis(self) -> float:
        """``κ₄/κ₂² = 1/(2n)`` --- independent of ``h``."""
        return 0.5 / self.n_dimensionless

    @property
    def rel_split(self) -> float:
        """``|γ − a| / max(γ, a)`` --- the dispatch coordinate of :func:`t_tilde`."""
        return abs(self.gamma - self.a) / max(self.gamma, self.a)

    def with_n(self, n: float) -> "ShotNoise":
        """Return a copy at dimensionless non-Gaussianity ``n``, ``κ₂`` fixed.

        Scales ``ν → n/(σ_t σ_x)`` and compensates ``h`` so that
        :attr:`variance` is unchanged --- so a sweep in ``n`` moves the
        non-Gaussian channels and *nothing else*.
        """
        nu_new = n / (self.sigma_t * self.sigma_x)
        h_new = self.h * np.sqrt(self.nu / nu_new)
        return ShotNoise(nu=nu_new, h=h_new, sigma_t=self.sigma_t,
                         sigma_x=self.sigma_x, gamma=self.gamma,
                         n_components=self.n_components)


#: Demo 3's headline parameter point: ``n = 1`` (skewness 0.63, comparable
#: to demo 2's 0.76), ``σ_t = 0.5`` and ``γ = 1`` so there is no scale
#: separation (contrast demo 2's ``σ_t = 0.3 ≪ 1/γ``) yet ``γ ≠ 1/σ_t``.
PARAMS = ShotNoise()
N_COMP = PARAMS.n_components


# =====================================================================
# Raw cumulants
# =====================================================================

def X_m(xs, sigma_x: float):
    r"""Spatial overlap ``X_m = ∫dx' Π_i e^{−|x_i − x'|/σ_x}``.

    Exact piecewise-analytic evaluation.  With the points sorted, the
    exponent ``S(x') = Σ_i |y_i − x'|`` is affine on each of the
    ``m − 1`` interior segments with integer slope ``q_j = 2j − m``, so
    each segment integrates in closed form (the ``q_j = 0`` segment,
    present only for even ``m``, degenerates to a rectangle).  The two
    outer tails contribute ``(σ_x/m) e^{−S/σ_x}`` each.

    Args:
        xs: ``(m, n)`` positions (or ``(m,)`` for a single point).
        sigma_x: pulse width.

    Returns:
        ``(n,)`` array.  Coincident points give ``2 σ_x / m``.
    """
    y = np.sort(_as_legs(xs), axis=0)
    m = y.shape[0]
    # S[j] = Σ_i |y_i − y_j|, the exponent at the j-th node.
    S = np.abs(y[None, :, :] - y[:, None, :]).sum(axis=1)
    out = (sigma_x / m) * (np.exp(-S[0] / sigma_x) + np.exp(-S[-1] / sigma_x))
    for j in range(1, m):
        q = 2 * j - m
        base = np.exp(-S[j - 1] / sigma_x)
        gap = y[j] - y[j - 1]
        if q == 0:
            out = out + gap * base
        else:
            # −(σ/q)·e^{−S_j−1/σ}·expm1(−qΔ/σ): stable as Δ → 0 and exact.
            out = out - (sigma_x / q) * base * np.expm1(-(q * gap) / sigma_x)
    return out


def T_m(ts, sigma_t: float):
    r"""Temporal overlap ``T_m = ∫ds Π_i Θ(t_i−s) e^{−(t_i−s)/σ_t}``.

    ``= (σ_t/m) · exp(−Σ_i (t_i − t_min)/σ_t)``; every exponent is
    non-positive, so this is stable at any ``t``.
    """
    t = _as_legs(ts)
    m = t.shape[0]
    return (sigma_t / m) * np.exp(-(t - t.min(axis=0)).sum(axis=0) / sigma_t)


def kappa_m(xs, ts, p: ShotNoise = PARAMS):
    """The raw ``m``-th cumulant amplitude ``ν h^m X_m T_m``.

    The full cumulant is ``δ_{a_1…a_m}`` times this (each component is an
    independent copy of the event process).
    """
    xs = _as_legs(xs)
    m = xs.shape[0]
    return p.nu * p.h ** m * X_m(xs, p.sigma_x) * T_m(ts, p.sigma_t)


# =====================================================================
# Response-contracted temporal kernel
# =====================================================================

def _as_legs(x):
    """Coerce ``(m,)`` (one sample) or ``(m, n)`` leg data to ``(m, n)``.

    ``np.atleast_2d`` would turn a length-``m`` vector into ``(1, m)`` ---
    i.e. silently reinterpret ``m`` legs as ``m`` samples of a 1-leg
    vertex --- so the leg axis is made explicit here instead.
    """
    x = np.asarray(x, dtype=float)
    return x[:, None] if x.ndim == 1 else x


def _phi1(z):
    """``(1 − e^{−z})/z``, with the removable singularity at ``z = 0``."""
    z = np.asarray(z, dtype=float)
    small = np.abs(z) < 1e-6
    safe = np.where(small, 1.0, z)
    return np.where(small, 1.0 - z / 2.0 + z * z / 6.0, -np.expm1(-safe) / safe)


def A_response(t, p: ShotNoise = PARAMS):
    r"""``A(t) = (e^{−a t} − e^{−γ t})/(γ − a) = J(t, 0)``.

    Written as ``t e^{−a t} φ₁((γ−a)t)`` so it is stable at ``γ = a``
    (where it becomes ``t e^{−a t}``).
    """
    t = np.asarray(t, dtype=float)
    return np.exp(-p.a * t) * t * _phi1((p.gamma - p.a) * t)


def J(t, s, p: ShotNoise = PARAMS):
    r"""``J(t', s) = ∫_0^{t'} R(t', u) g(u − s) du``.

    Three regimes, because the noise is stationary from the infinite past
    while the field starts at ``t = 0``::

        s > t' :  0
        0 ≤ s  :  (e^{−aτ} − e^{−γτ})/(γ − a),  τ = t' − s
        s < 0  :  e^{a s} · A(t')
    """
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    tau = np.maximum(t - s, 0.0)
    inside = np.exp(-p.a * tau) * tau * _phi1((p.gamma - p.a) * tau)
    before = np.exp(p.a * np.minimum(s, 0.0)) * A_response(t, p)
    return np.where(s > t, 0.0, np.where(s < 0.0, before, inside))


def t_tilde_closed(ts, p: ShotNoise = PARAMS, return_error: bool = False):
    r"""``T̃_m`` by the exact ``2^m``-term closed form.

    Splitting the ``s`` integral at 0 and expanding
    ``Π_i (e^{−a(t_i−s)} − e^{−γ(t_i−s)})`` over subsets ``S`` of the legs::

        T̃_m = Π_i A(t_i)/(m a)
             + (γ−a)^{−m} Σ_S (−1)^{m−|S|} (e^{−E_S} − e^{−c_S}) / b_S,
        b_S = a|S| + γ(m−|S|),
        c_S = a Σ_{i∈S} t_i + γ Σ_{i∉S} t_i,
        E_S = c_S − b_S T   with  T = min_i t_i,

    and *every* exponent is non-positive (``E_S`` is written relative to
    ``T``), so the expression is stable at arbitrarily large ``t``.

    The ``(γ−a)^{−m}`` prefactor multiplies a sum that vanishes to order
    ``(γ−a)^m``: exact in exact arithmetic, but losing digits in floating
    point.  ``return_error=True`` additionally returns a per-sample
    estimate of that loss, ``ε_machine · Σ_S |term_S| / |result|`` --- the
    cancellation amplification factor is directly computable, so
    :func:`t_tilde` never has to guess.  Verified against 60-digit
    ``mpmath`` in ``tests/test_demo3_shot_noise.py``.
    """
    eps = p.gamma - p.a
    t = _as_legs(ts)
    m = t.shape[0]
    T = t.min(axis=0)
    out = np.prod(A_response(t, p), axis=0) / (m * p.a)
    acc = np.zeros_like(T)
    acc_abs = np.zeros_like(T)
    for mask in itertools.product((False, True), repeat=m):
        sel = np.array(mask)
        k = int(sel.sum())
        b = p.a * k + p.gamma * (m - k)
        c = p.a * t[sel].sum(axis=0) + p.gamma * t[~sel].sum(axis=0)
        E = p.a * (t[sel] - T).sum(axis=0) + p.gamma * (t[~sel] - T).sum(axis=0)
        e_E, e_c = np.exp(-E), np.exp(-c)
        term = (e_E - e_c) / b
        acc = acc + (-1.0) ** (m - k) * term
        # Two independent error sources per term: the cancellation in the
        # signed sum (captured by |term|), and the fact that ``exp(−E)``
        # inherits a relative error ~ε|E| from rounding in its *argument*
        # -- and E reaches ~80 at widely-spread t'.  Weight the exponent
        # amplification by which of the two exponentials actually
        # survives, so the common case (E = 0, e^{−c} negligible) picks up
        # no spurious pessimism.
        denom = np.maximum(e_E + e_c, np.finfo(float).tiny)
        amp = (E * e_E + c * e_c) / denom
        acc_abs = acc_abs + (1.0 + amp) * np.abs(term)
    # eps == 0 (γ = 1/σ_t exactly) is a *removable* singularity: the 0/0
    # is expected, produces nan, and is routed to the quadrature branch by
    # :func:`t_tilde`.  Silence the warning rather than let it surface.
    with np.errstate(divide="ignore", invalid="ignore"):
        value = out + acc / eps ** m
        if not return_error:
            return value
        scale = np.maximum(np.abs(value), np.finfo(float).tiny)
        est = _EPS * (acc_abs / abs(eps) ** m + np.abs(out)) / scale
    return value, est


def _panel_edges(m: int, p: ShotNoise) -> np.ndarray:
    """Geometrically graded panel edges in ``v = T − s``, from 0 outwards.

    The integrand decays at least as fast as ``e^{−m·min(a,γ)·v}``; panels
    double from ``0.5/(m·max(a,γ))`` until :data:`DECADES` e-foldings are
    covered.
    """
    delta = 0.5 / (m * max(p.a, p.gamma))
    v_max = DECADES / (m * min(p.a, p.gamma))
    # ``arange(n_pan)`` tops out at ``delta·2^(n_pan−1)``, so the ``+ 1``
    # is what makes the stack actually *reach* ``v_max``.  Without it the
    # tail beyond the last panel is dropped, which costs ~2e-8 relative at
    # large ``T`` and does NOT improve with more nodes per panel --- the
    # signature that told us this was truncation, not quadrature order.
    n_pan = max(1, int(np.ceil(np.log2(max(v_max / delta, 2.0)))) + 1)
    return np.concatenate([[0.0], delta * 2.0 ** np.arange(n_pan)])


def t_tilde_quad(ts, p: ShotNoise = PARAMS, n_gl: int = N_GL_PANEL):
    r"""``T̃_m`` by graded composite Gauss-Legendre in the source time ``s``.

    The ``s < 0`` half is done analytically (``Π_i A(t_i)/(m a)``, no
    cancellation); the ``s ∈ [0, T]`` half is a 1-D integral of a smooth
    product of ``J``'s, so a geometrically graded panel stack converges
    exponentially.  Uniformly stable --- in particular at ``γ = 1/σ_t``,
    where :func:`t_tilde_closed` is 0/0.

    Panels are clipped *per sample* at ``v = T`` (recomputing the affine
    map), not by zeroing weights: the latter mistreats the straddling
    panel and costs ~10 orders of accuracy.
    """
    t = _as_legs(ts)
    m = t.shape[0]
    T = t.min(axis=0)
    out = np.prod(A_response(t, p), axis=0) / (m * p.a)
    x, w = leggauss(n_gl)
    edges = _panel_edges(m, p)
    acc = np.zeros_like(T)
    for lo, hi in zip(edges[:-1], edges[1:]):
        lo_c, hi_c = np.minimum(lo, T), np.minimum(hi, T)
        half, mid = 0.5 * (hi_c - lo_c), 0.5 * (hi_c + lo_c)
        s = T[:, None] - (mid[:, None] + half[:, None] * x[None, :])
        prod = np.ones_like(s)
        for i in range(m):
            prod = prod * J(t[i][:, None], s, p)
        acc = acc + half * np.sum(w[None, :] * prod, axis=1)
    return out + acc


def t_tilde(ts, p: ShotNoise = PARAMS, method: str = "auto",
            tol: float = AUTO_TOL):
    """``T̃_m(t')`` --- the R-contracted temporal kernel.

    Args:
        ts: ``(m, n)`` partner (outer) times.
        p: model parameters.
        method: ``'auto'`` (default) evaluates the closed form together
            with its conditioning estimate and recomputes *only* the
            samples whose estimate exceeds ``tol`` by quadrature;
            ``'closed'`` / ``'quad'`` force a branch.
        tol: relative-error budget for the ``'auto'`` dispatch.

    The auto path costs one closed-form evaluation plus, in the rare bad
    corners, a quadrature over the offending subset --- so it is as fast
    as the closed form wherever the closed form is trustworthy, and as
    accurate as the quadrature everywhere else.
    """
    if method == "closed":
        return t_tilde_closed(ts, p)
    if method == "quad":
        return t_tilde_quad(ts, p)
    if method != "auto":
        raise ValueError(f"method must be 'auto', 'closed' or 'quad', got {method!r}")
    value, est = t_tilde_closed(ts, p, return_error=True)
    bad = ~np.isfinite(value) | (est > tol)
    if np.any(bad):
        value = np.array(value, dtype=float, copy=True)
        value[bad] = t_tilde_quad(_as_legs(ts)[:, bad], p)
    return value


def K_R(xs, ts, p: ShotNoise = PARAMS, method: str = "auto"):
    r"""The R-contracted cumulant ``K_R = ν h^m X_m(x') T̃_m(t')``.

    The drift has no spatial derivative, so ``R`` is diagonal in space and
    the leg positions coincide with the partner positions ``x'`` --- the
    spatial factor comes through untouched.

    With ``F = 0`` this *is* the exact ``m``-point function of the field:
    ``⟨φ(x'_1,t'_1) … φ(x'_m,t'_m)⟩_c = K_R``.
    """
    xs = _as_legs(xs)
    m = xs.shape[0]
    return p.nu * p.h ** m * X_m(xs, p.sigma_x) * t_tilde(ts, p, method)


# =====================================================================
# κ² for the package's built-in closed-form C
# =====================================================================

def kappa2_lam(p: ShotNoise = PARAMS) -> float:
    """Amplitude for ``ExponentialTemporal``: ``λ = ν h² σ_t / 2``.

    ``κ²(Δt, r) = ν h² (σ_t/2) e^{−|Δt|/σ_t} · X_2(r)``, i.e. exactly
    ``ExponentialTemporal(lam=λ, sigma_t=σ_t) × CustomKernel(X_2)`` ---
    a ``SeparableTranslation`` with an exponential temporal kernel, which
    is precisely the family ``builtin_closed_form_for`` recognises.  No
    ``C`` quadrature ever runs.
    """
    return p.nu * p.h ** 2 * p.sigma_t / 2.0


def kappa2_spatial(r, p: ShotNoise = PARAMS):
    """``X_2(r) = σ_x (1 + r/σ_x) e^{−r/σ_x}`` --- the spatial envelope of κ².

    Note this is *not* exponential: convolving two exponential pulses
    gives the extra linear factor.  The built-in closed form accepts any
    spatial callable, so this costs nothing.
    """
    r = np.abs(np.asarray(r, dtype=float))
    return p.sigma_x * (1.0 + r / p.sigma_x) * np.exp(-r / p.sigma_x)


# =====================================================================
# NonLocalVertex couplings (already_R_contracted=True)
# =====================================================================

def _diagonal_tensor(amp: np.ndarray, m: int, n_comp: int) -> np.ndarray:
    """Embed ``(n,)`` amplitudes on the ``δ_{a_1…a_m}`` diagonal of ``(n,)+(N,)*m``."""
    out = np.zeros((amp.shape[0],) + (n_comp,) * m)
    for a in range(n_comp):
        out[(slice(None),) + (a,) * m] = amp
    return out


def _coupling_vectorized(n_2d, t_2d, m: int, p: ShotNoise):
    n = np.asarray(n_2d, dtype=float)
    t = np.asarray(t_2d, dtype=float)
    if n.shape[0] != m or t.shape[0] != m:
        raise ValueError(f"expected {m} legs, got n{n.shape} t{t.shape}")
    return _diagonal_tensor(K_R(n, t, p), m, p.n_components)


def coupling_vectorized_for(m: int):
    """Return a batched ``already_R_contracted`` κ^(m) callable for any ``m``.

    The construction is **m-agnostic**: the ``s``-factorisation of the
    filtered-Poisson cumulants and the ``2^m`` expansion of ``T̃_m`` work
    for every order with no new ideas.  That is what makes the neglected
    cumulant ladder (κ⁵ and beyond) *computable* here rather than merely
    boundable.
    """
    def _fn(n_2d, t_2d, p: ShotNoise = PARAMS, _m=m):
        return _coupling_vectorized(n_2d, t_2d, _m, p)
    _fn.__name__ = f"coupling_k{m}_vectorized"
    _fn.__qualname__ = _fn.__name__
    return _fn


def kappa_ratio(m: int, p: ShotNoise = PARAMS) -> float:
    """``κ_m / κ₃`` at coincident points ``= h^{m−3} (3/m)²``.

    The neglected-cumulant ladder in closed form.  ``h`` is a free
    parameter (it sets the field's units, not its shape --- the skewness
    and excess kurtosis depend on ``n`` alone), so the ladder ratio can
    be made small by choice without touching the non-Gaussianity.
    """
    return p.h ** (m - 3) * (3.0 / m) ** 2


def coupling_k3_vectorized(n_2d, t_2d, p: ShotNoise = PARAMS):
    """Batched ``already_R_contracted`` κ³: ``(3, n) → (n, N, N, N)``."""
    return _coupling_vectorized(n_2d, t_2d, 3, p)


def coupling_k4_vectorized(n_2d, t_2d, p: ShotNoise = PARAMS):
    """Batched ``already_R_contracted`` κ⁴: ``(4, n) → (n, N, N, N, N)``."""
    return _coupling_vectorized(n_2d, t_2d, 4, p)


def coupling_k3(n_list, t_list, p: ShotNoise = PARAMS):
    """Per-sample ``already_R_contracted`` κ³: length-3 sequences → ``(N,N,N)``."""
    n = np.asarray(n_list, dtype=float)[:, None]
    t = np.asarray(t_list, dtype=float)[:, None]
    return coupling_k3_vectorized(n, t, p)[0]


def coupling_k4(n_list, t_list, p: ShotNoise = PARAMS):
    """Per-sample ``already_R_contracted`` κ⁴: length-4 sequences → ``(N,)*4``."""
    n = np.asarray(n_list, dtype=float)[:, None]
    t = np.asarray(t_list, dtype=float)[:, None]
    return coupling_k4_vectorized(n, t, p)[0]


# =====================================================================
# Raw (un-contracted) κ^(m) couplings -- for the raw-vs-R-contracted test
# =====================================================================

def coupling_k3_raw_vectorized(n_2d, t_2d, p: ShotNoise = PARAMS):
    """Batched **raw** κ³ at the vertex legs: ``(3, n) → (n, N, N, N)``.

    ``already_R_contracted=False`` companion of :func:`coupling_k3_vectorized`;
    the runtime then does the three leg integrals itself.
    """
    n = np.asarray(n_2d, dtype=float)
    t = np.asarray(t_2d, dtype=float)
    return _diagonal_tensor(kappa_m(n, t, p), 3, p.n_components)


def coupling_k4_raw_vectorized(n_2d, t_2d, p: ShotNoise = PARAMS):
    """Batched **raw** κ⁴ at the vertex legs: ``(4, n) → (n, N, N, N, N)``."""
    n = np.asarray(n_2d, dtype=float)
    t = np.asarray(t_2d, dtype=float)
    return _diagonal_tensor(kappa_m(n, t, p), 4, p.n_components)
