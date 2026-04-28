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
            interp_method: ``RegularGridInterpolator`` method used by
                full-grid C tables. ``'linear'`` (default) is monotone
                and safe for steep cosmological tails; ``'cubic'``
                gives O(h⁴) accuracy on smooth, well-sampled grids.
                See :class:`PropagatorCache` docstring for the full
                list of accepted methods and the linear-vs-cubic
                trade-off.
        """
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
            "interp_method": interp_method,
        }

        def _build() -> "Propagators":
            model = system.build_propagator_model()
            if c_closed_form is not None:
                cache = _ClosedFormPropagatorCache(
                    model=model, homogeneity=hom, c_fn=c_closed_form,
                    interp_method=interp_method,
                )
            else:
                cache = PropagatorCache(
                    model=model, homogeneity=hom,
                    interp_method=interp_method,
                )

            is_lazy = False
            if hom == "translation":
                cache.precompute_C_table_translation(
                    t_max=t_max, n_grid_t=n_grid_t,
                    r_max=r_max, n_grid_r=n_grid_r,
                    n_jobs=n_jobs,
                )
                is_lazy = (r_max is None) or (n_grid_r is None)
            elif hom == "rotation":
                cache.precompute_C_table_rotation(
                    t_max=t_max, n_grid_t=n_grid_t,
                    n_grid_cos=n_grid_cos,
                    n_jobs=n_jobs,
                )
                is_lazy = n_grid_cos is None
            else:  # general
                cache.precompute_C_table_general(
                    t_max=t_max, n_grid_t=n_grid_t,
                    x_max=x_max, n_grid_x=n_grid_x,
                    n_jobs=n_jobs,
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

    def _C_value_direct(self, n1, t1, n2, t2):
        return np.asarray(self._c_fn(n1, t1, n2, t2))


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
