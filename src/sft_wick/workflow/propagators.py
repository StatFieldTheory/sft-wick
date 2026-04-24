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
        }

        def _build() -> "Propagators":
            model = system.build_propagator_model()
            cache_cls = (
                _make_closed_form_cache_cls(c_closed_form)
                if c_closed_form is not None
                else PropagatorCache
            )
            cache = cache_cls(model=model, homogeneity=hom)

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

            return cls(cache=cache, homogeneity=hom, is_lazy=is_lazy)

        return load_or_compute(
            cache_path, spec_key, _build,
            operation_name="propagator table",
        )


def _make_closed_form_cache_cls(c_fn: Callable):
    """Build a ``PropagatorCache`` subclass whose ``_C_value_direct``
    returns ``c_fn(n1, t1, n2, t2)``.

    This is the clean in-library analogue of the ``_FastCache``
    subclass used by ``tests/test_deductive_numerics.py`` and
    ``examples/demo1/validate_phase5.py`` — it short-circuits the
    ``dblquad`` in every spline-table builder to the user-supplied
    closed form.  Result: spatial spline builds complete in
    milliseconds instead of minutes.
    """

    class _ClosedFormC(PropagatorCache):
        def _C_value_direct(self, n1, t1, n2, t2):
            return np.asarray(c_fn(n1, t1, n2, t2))

    _ClosedFormC.__name__ = "_ClosedFormC"
    _ClosedFormC.__qualname__ = _ClosedFormC.__name__
    return _ClosedFormC


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
