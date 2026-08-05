"""``Propagators`` — thin wrapper around :class:`PropagatorCache` that
auto-dispatches precompute to the right homogeneity builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from sft_wick.evaluate import PropagatorCache


@dataclass(frozen=True)
class Propagators:
    """Holds a :class:`PropagatorCache` preconfigured for a
    :class:`~sft_wick.workflow.System`.  Opaque to the user — the
    only thing they do with this object is pass it to
    :meth:`Expansion.evaluate` / :meth:`Expansion.sweep`.

    Attributes:
        cache: the underlying :class:`PropagatorCache`.
        homogeneity: resolved homogeneity string.
        is_lazy: whether the cache is in lazy-spline mode for its
            spatial dimension.
    """

    cache: PropagatorCache
    homogeneity: str
    is_lazy: bool

    @classmethod
    def build(
        cls,
        system,
        *,
        t_max: float,
        n_grid_t: int = 60,
        homogeneity: str | None = None,
        r_max: float | None = None,
        n_grid_r: int | None = None,
        n_grid_cos: int | None = None,
        x_max: float | None = None,
        n_grid_x: int | None = None,
        n_jobs: int = 1,
        c_closed_form: Callable | None = None,
        cache_path: Any = None,
        interp_method: str = "linear",
        c_closed_form_only: bool = False,
        c_closed_form_vectorized: bool = False,
        c_method: str = "dblquad",
        c_n_gauss: int = 20,
        diag_C: bool = True,
    ) -> "Propagators":
        """Construct a ``Propagators`` for ``system``.  Called
        indirectly via :meth:`System.propagators`.

        Args:
            c_closed_form: optional fast path for C evaluation.  If
                provided, it must be a callable
                ``(n1, t1, n2, t2) -> (N, N)`` returning the full C
                matrix at that spacetime-pair — the same signature
                as :meth:`PropagatorCache._C_value_direct`.  When
                set, the wrapper builds a :class:`PropagatorCache`
                subclass that uses it instead of ``dblquad``,
                collapsing the spline-table build time from minutes
                (typical for ``scipy.integrate.dblquad`` on fine
                grids) to milliseconds.  Intended for kernels with
                known closed-form C (OU, separable exponentials).
            c_closed_form_only: when True (and ``c_closed_form`` is
                provided), skip the spline interpolator entirely.
                ``cache.C_at_batch`` then routes every lookup
                straight through the user's c_fn -- machine-precision
                agreement with the analytical C, no truncation
                error from grid spacing or spline order. Use this
                when the closed form is fast and the spline error
                would dominate over QMC noise (e.g. demo1's OU
                kernel where ``sigma_t = 0.3`` requires ``dt < 0.1``
                for sub-percent spline accuracy).
            c_closed_form_vectorized: only meaningful when
                ``c_closed_form_only=True``. When True, the user's
                c_fn must accept batched ``(t1, t2, x1, x2)`` arrays
                of shape ``(n,)`` and return a ``(n, N, N)`` tensor
                in a single call. When False, the cache falls back
                to a Python per-sample loop (slow; useful only for
                small point evaluations or when migrating an
                existing scalar c_fn).
            interp_method: ``RegularGridInterpolator`` method used by
                full-grid C tables. ``'linear'`` (default) is monotone
                and safe for steep cosmological tails; ``'cubic'``
                gives O(h⁴) accuracy on smooth, well-sampled grids.
                See :class:`PropagatorCache` docstring for the full
                list of accepted methods and the linear-vs-cubic
                trade-off. Ignored under ``c_closed_form_only=True``.
            c_method: How the inner 2-D C-propagator integral
                ``∫ R κ² R`` is evaluated when the cache builds its
                table.

                - ``'dblquad'`` (default) -- ``scipy.integrate.dblquad``
                  adaptive Gauss-Kronrod, robust on any κ² but slow
                  (10-80 ms / call).
                - ``'gauss_legendre'`` -- tensor-product GL with a
                  diagonal-aware sub-region split at ``λ1 = λ2``.
                  18-100× faster on κ² that is **piecewise analytic**
                  with at most a single ``|λ1−λ2|`` cusp on the
                  diagonal (the standard exponential / Gaussian / OU
                  family used in demo1, demo2, and the test suite).
                  Returns near-machine-precision agreement with
                  ``'dblquad'`` at ``c_n_gauss=20``.

                Ignored when ``c_closed_form_only=True`` (the C
                lookup bypasses the cache entirely in that mode).
            c_n_gauss: Per-dimension GL node count for
                ``c_method='gauss_legendre'`` (default 20, enough for
                machine precision on smooth OU / Gaussian kernels).
                Cost scales as ``c_n_gauss²`` per sub-region.
        """
        if c_closed_form_only and c_closed_form is None:
            raise ValueError(
                "c_closed_form_only=True requires c_closed_form to be "
                "set (the no-spline path needs a closed-form callable "
                "to use as the lookup function)."
            )
        if c_closed_form_vectorized and not c_closed_form_only:
            raise ValueError(
                "c_closed_form_vectorized=True only makes sense with "
                "c_closed_form_only=True; the vectorised contract is "
                "specific to the no-spline lookup path."
            )
        # diag_C=False propagates the full (N, N) matrix returned by
        # c_fn through the integrator. The spline-table builders
        # (precompute_C_table_* in evaluate.py) only fill diagonal
        # entries, so off-diagonal preservation is meaningless without
        # the closed-form-only path. Reject the combination early to
        # avoid silently dropping off-diagonals at lookup time.
        if (not diag_C) and (not c_closed_form_only):
            raise ValueError(
                "diag_C=False requires c_closed_form_only=True. "
                "Spline-table paths only build diagonal C entries; "
                "off-diagonal observables (e.g. kappa-gamma cross "
                "correlations) need the closed-form-only path."
            )
        from .cache import load_or_compute

        hom = homogeneity if homogeneity is not None else system.homogeneity
        if hom not in ("translation", "rotation", "general"):
            raise ValueError(
                f"homogeneity must be one of 'translation', 'rotation', "
                f"'general'; got {hom!r}."
            )

        # Spec key = all inputs that affect the built cache content.
        spec_key = {
            "system_hash": _minimal_propagator_spec(system),
            "hom": hom,
            "t_max": t_max,
            "n_grid_t": n_grid_t,
            "r_max": r_max, "n_grid_r": n_grid_r,
            "n_grid_cos": n_grid_cos,
            "x_max": x_max, "n_grid_x": n_grid_x,
            "c_closed_form_repr":
                None if c_closed_form is None else repr(c_closed_form),
            "c_closed_form_only": c_closed_form_only,
            "c_closed_form_vectorized": c_closed_form_vectorized,
            "interp_method": interp_method,
            "c_method": c_method,
            "c_n_gauss": int(c_n_gauss),
            "diag_C": bool(diag_C),
        }

        def _build() -> "Propagators":
            model = system.build_propagator_model(diag_C=diag_C)
            if c_closed_form is not None:
                cache = _ClosedFormPropagatorCache(
                    model=model, homogeneity=hom, c_fn=c_closed_form,
                    interp_method=interp_method,
                    c_method=c_method, n_gauss=int(c_n_gauss),
                )
            else:
                cache = PropagatorCache(
                    model=model, homogeneity=hom,
                    interp_method=interp_method,
                    c_method=c_method, n_gauss=int(c_n_gauss),
                )

            if c_closed_form_only:
                # Skip every spline build. C_at_batch uses
                # ``_closed_form_at_batch_diag`` from now on, which
                # routes lookups straight through the user's c_fn.
                cache._closed_form_only = True
                cache._closed_form_vectorized = c_closed_form_vectorized
                return cls(cache=cache, homogeneity=hom, is_lazy=False)

            is_lazy = False
            if hom == "translation":
                cache.precompute_C_table_translation(
                    t_max=t_max, n_grid_t=n_grid_t,
                    r_max=r_max, n_grid_r=n_grid_r,
                    n_jobs=n_jobs,
                    c_method=c_method, n_gauss=int(c_n_gauss),
                )
                is_lazy = (r_max is None) or (n_grid_r is None)
            elif hom == "rotation":
                cache.precompute_C_table_rotation(
                    t_max=t_max, n_grid_t=n_grid_t,
                    n_grid_cos=n_grid_cos,
                    n_jobs=n_jobs,
                    c_method=c_method, n_gauss=int(c_n_gauss),
                )
                is_lazy = n_grid_cos is None
            else:  # general
                cache.precompute_C_table_general(
                    t_max=t_max, n_grid_t=n_grid_t,
                    x_max=x_max, n_grid_x=n_grid_x,
                    n_jobs=n_jobs,
                    c_method=c_method, n_gauss=int(c_n_gauss),
                )
                is_lazy = (x_max is None) or (n_grid_x is None)

            # Pin lazy-cache n_jobs to 1. _LazyTimeSplineCache._build is
            # triggered from inside QMC sampling when a worker hits a new
            # parameter value; if Layer 2 (integrate_diagrams) or Layer 3
            # (Expansion.sweep) is itself parallel, an inner Parallel(...)
            # call here would spawn a nested loky pool. Lazy builds are
            # n_grid_t**2 independent _C_value_direct calls and typically
            # account for a small fraction of total wall-time, so we let
            # the outer parallelism saturate the cores instead.
            for lazy_attr in (
                "_lazy_translation",
                "_lazy_rotation",
                "_lazy_general",
            ):
                lazy = getattr(cache, lazy_attr, None)
                if lazy is not None:
                    lazy.n_jobs = 1

            return cls(cache=cache, homogeneity=hom, is_lazy=is_lazy)

        return load_or_compute(
            cache_path, spec_key, _build,
            operation_name="propagator table",
        )


def propagators_from_cache(cache) -> "Propagators":
    """Wrap a hand-built :class:`~sft_wick.evaluate.PropagatorCache` as a
    :class:`Propagators`, so it can be passed to
    :meth:`~sft_wick.workflow.Expansion.evaluate` / ``sweep``.

    ``System.propagators()`` builds a cache FROM the system's own model, which
    is the right default -- but it gives no supported way to use a cache that
    does not come from a ``System`` at all.  The disorder-averaged
    :func:`sft_wick.spectral.spectral_cache` is exactly that: its ``R`` and
    ``C`` come from a spectrum, not from the system's ``kappa2``.

    (``Propagators`` is a plain dataclass, so a determined caller could always
    have constructed one by hand; what this adds is a supported, named entry
    point -- not a capability that was absent.)

    The component count is NOT checked here, because a cache carries no record
    of which system it was meant for.  It IS checked at use, in
    :meth:`Expansion.evaluate`, where both numbers are known.
    """
    return Propagators(cache=cache, homogeneity="translation", is_lazy=False)



class _ClosedFormPropagatorCache(PropagatorCache):
    """PropagatorCache that delegates ``_C_value_direct`` to a user callable.

    Defined at module level (not inside a function) so that joblib's loky
    workers can re-import the class when distributing per-cell tasks across
    subprocesses. The user ``c_fn`` is held as an instance attribute and
    is itself loaded by :func:`_load_callable_from_module` under the
    ``.py`` file's bare basename, so it is round-trippable across workers.

    This replaces the earlier ``_make_closed_form_cache_cls`` factory which
    returned a class defined inside a function and was therefore not
    transportable across loky boundaries — forcing ``n_jobs = 1``.
    """

    def __init__(self, *args, c_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        if c_fn is None:
            raise ValueError(
                "_ClosedFormPropagatorCache requires a c_fn callable."
            )
        self._c_fn = c_fn

    def _C_value_direct(self, n1, t1, n2, t2, **_quad_kwargs):
        # Accept (and ignore) the ``method=`` / ``n_gauss=`` quadrature
        # kwargs that ``precompute_C_table_*`` may forward when the
        # caller requests ``c_method='gauss_legendre'``.  In closed-form
        # mode the user-supplied callable is exact, so neither dblquad
        # nor GL is invoked here.
        arr = np.asarray(self._c_fn(n1, t1, n2, t2))
        # When the user supplies a vectorised c_fn (returns
        # (n, N, N)) but a legacy single-point caller hands in
        # scalar t1/t2, the result is a 3-D array with a length-1
        # leading axis. Squeeze only for true scalar-time calls so the
        # batched no-spline path can keep its required ``(1, N, N)``
        # shape when t1/t2 arrive as length-1 arrays.
        scalar_time = np.asarray(t1).ndim == 0 and np.asarray(t2).ndim == 0
        if scalar_time and arr.ndim == 3 and arr.shape[0] == 1:
            return arr[0]
        return arr


def _minimal_propagator_spec(system) -> Any:
    """Lightweight spec key for propagator-cache hashing.  Includes
    only the fields that actually determine the cache content (not
    the interaction vertices)."""
    return {
        "n_components": system.n_components,
        "t_min": system.t_min,
        "iso_R": system.iso_R,
        "linear_repr": repr(system.linear),
        "noise_kappa2_repr": repr(system.noise.kappa2),
        "noise_sigma2_repr": repr(system.noise.sigma2),
    }
