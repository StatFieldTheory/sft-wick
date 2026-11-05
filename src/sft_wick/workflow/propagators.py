"""``Propagators``: thin wrapper around :class:`PropagatorCache` that
auto-dispatches precompute to the right homogeneity builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sft_wick.evaluate import (
    C_METHODS,
    DEFAULT_C_N_GAUSS,
    DEFAULT_N_GRID_T,
    HOMOGENEITIES,
    PropagatorCache,
)

from ._messages import shown
from .specs import _flag


@dataclass(frozen=True)
class Propagators:
    """Holds a :class:`PropagatorCache` preconfigured for a
    :class:`~sft_wick.workflow.system.System`.  Opaque to the user: the
    only thing they do with this object is pass it to
    :meth:`Expansion.evaluate` / :meth:`Expansion.sweep`.

    Attributes:
        cache: the underlying :class:`PropagatorCache`.
        homogeneity: resolved homogeneity string.
        is_lazy: whether the cache is in lazy-spline mode for its
            spatial dimension.
        c_source: how C is evaluated, after the ``'auto'`` choices were
            resolved: ``'closed_form:builtin'``, ``'closed_form:user'``,
            ``'quadrature:gauss_legendre(n=20)'``,
            ``'quadrature:graded(levels=12)'``, ``'quadrature:dblquad'`` or
            ``'quadrature:per_position'``, as of the build.  This is what the
            CLI banner and ``--dry-run`` report.  A table's pairs of positions
            each have a rule: the label names the largest count or depth, and
            ``per_position`` when only some pairs take ``dblquad``.  A lazy
            table has only the origin to report, or ``per_position`` when the
            origin settles nothing, and it can use a finer rule at a position
            later; ``cache.c_method_resolved`` follows the positions read.
    """

    cache: PropagatorCache
    homogeneity: str
    is_lazy: bool
    c_source: str = "unknown"

    @classmethod
    def build(
        cls,
        system,
        *,
        t_max: float,
        n_grid_t: int = DEFAULT_N_GRID_T,
        homogeneity: str | None = None,
        r_max: float | None = None,
        n_grid_r: int | None = None,
        n_grid_cos: int | None = None,
        x_max: float | None = None,
        n_grid_x: int | None = None,
        n_jobs: int = 1,
        c_closed_form: Callable | str | None = "auto",
        cache_path: Any = None,
        interp_method: str = "linear",
        c_closed_form_only: bool = False,
        c_closed_form_vectorized: bool = False,
        c_method: str = "auto",
        c_n_gauss: int = DEFAULT_C_N_GAUSS,
        diag_C: bool = True,
    ) -> "Propagators":
        """Construct a ``Propagators`` for ``system``.
        :meth:`System.propagators` is the usual route; this method holds
        its refusals (under ``Raises``), so a direct call has them too.

        Args:
            c_closed_form: how C is obtained without quadrature.

                - ``'auto'`` (default): use the built-in closed form
                  (:func:`~sft_wick.workflow.closed_forms.builtin_closed_form_for`)
                  when the system's kernel family has one (diagonal
                  constant drift + separable translation-invariant noise
                  with an exponential temporal kernel, optionally a
                  constant white-noise impulse), and fall through to
                  quadrature (``c_method``) otherwise.
                - ``None``: never use the built-in form; always run
                  quadrature.
                - a callable ``(n1, t1, n2, t2) -> (N, N)`` returning the
                  full C matrix at that spacetime pair (the signature of
                  :meth:`PropagatorCache._C_value_direct`): a user
                  closed form, which takes precedence over the built-in
                  one.  Under ``c_closed_form_only=True`` it may also
                  be batched (``(n,)`` times → ``(n, N, N)``); the
                  built-in form supports both contracts.  A
                  ``has_diagonal_kink`` attribute says whether it has a
                  derivative jump on ``t1 = t2``; without one the
                  integrators that split probe it for the jump.

                Either closed form cuts the spline-table build from
                minutes (``dblquad`` on a fine grid) to milliseconds.
            c_closed_form_only: when True (and ``c_closed_form`` is
                provided), skip the spline interpolator entirely.
                ``cache.C_at_batch`` then routes every lookup
                straight through the user's c_fn, giving machine-precision
                agreement with the analytical C and no truncation
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
                ``∫ R κ² R`` is evaluated when no closed form applies.

                - ``'auto'`` (default): for every kernel, the package's
                  and a user callable alike (``GeneralKappa2``,
                  ``CustomKernel``, ``CustomImpulse``, ``ExplicitR``,
                  callable ``gamma``), Gauss-Legendre with the node count
                  chosen by :func:`~sft_wick.evaluate.select_gl_node_count`,
                  refined from ``c_n_gauss`` until the rule agrees with
                  itself at the table's probe cells.  When no count
                  converges, a kernel or a rate far narrower than the
                  horizon, the rule is a Gauss-Legendre rule on panels
                  halved towards the ends of each interval
                  (:func:`~sft_wick.evaluate.select_graded_levels`), with
                  ``dblquad`` and a warning when that does not converge
                  either, as for a kink or a step inside the interval.
                  ``c_source`` reports the rule chosen.  A kernel that is
                  not separable has a rule per pair of positions, of a full
                  grid or of a lazy table as it reads them, and one pair
                  with no converged rule takes ``dblquad`` alone, with a
                  warning.  ``c_source`` names the largest count or depth of
                  a full grid, or ``quadrature:per_position`` when only some
                  pairs take ``dblquad``.  On a lazy table it is the rule at
                  the origin, or ``quadrature:per_position`` when the kernel
                  raises at the origin, is not finite there or vanishes
                  there, or, on a kernel that is not separable, no rule
                  converges there.  The check compares two rules: a feature narrower than the
                  node spacing that both miss passes it, and that includes
                  a spike on the diagonal narrower than about ``1e-8`` of
                  the horizon, for which white noise is the spelling.
                  ``c_n_gauss`` is where the refinement starts and is
                  tested at least once however large.
                - ``'gauss_legendre'``: tensor-product GL with a
                  diagonal-aware sub-region split at ``λ1 = λ2`` and
                  exactly ``c_n_gauss`` nodes, no convergence check.
                  18-100× faster than dblquad on κ² that is **piecewise
                  analytic** with at most a single ``|λ1−λ2|`` cusp on
                  the diagonal; accuracy at fixed ``c_n_gauss`` degrades
                  with ``(γ + 1/σ_t) t_max`` (see
                  :class:`~sft_wick.evaluate.PropagatorCache`).
                - ``'dblquad'``: ``scipy.integrate.dblquad`` adaptive
                  Gauss-Kronrod, 10-250 ms per call, with no convergence
                  check: a kernel far narrower than the interval can come
                  back wrong by any factor.

                Ignored when a closed form is in use.
            c_n_gauss: Per-dimension GL node count for
                ``c_method='gauss_legendre'``; under ``'auto'`` only the
                count the refinement starts from (default 20).  Cost scales as
                ``c_n_gauss²`` per sub-region.
            diag_C: ``True`` (default): the tables hold ``C_aa`` and C
                lookups return ``(n, N)`` diagonals, and every read of a
                closed form takes its diagonal.  ``False``: every
                table holds all ``N²`` entries ``C_ab``, by quadrature or
                from the closed form (which must then return the full
                matrix), and lookups return ``(n, N, N)``.  Needed when C
                has off-diagonal entries: a dense R, a component-mixing
                κ², a matrix σ².  A Gauss-Legendre cell costs the same
                either way, since it contracts ``R κ² Rᵀ`` as matrices; a
                ``dblquad`` cell runs ``N²`` adaptive integrals instead of
                ``N``.  When a built-in even temporal kernel with no σ²
                gives ``C_ab(t1, t2) = C_ba(t2, t1)``, half of each table is
                filled by that relation.  A custom kernel gets every cell.

        Raises:
            ValueError: when ``t_max`` exceeds ``t_max_cache`` of a
                :class:`DiagonalA` with a callable rate; with
                ``diag_C=True``, when C has off-diagonal entries.
        """
        diag_C = _flag(diag_C, "diag_C", "Propagators.build")
        system._require_buildable(t_max, diag_C)
        c_fn, c_fn_source = _resolve_closed_form_flags(
            system, c_closed_form, c_closed_form_only, c_closed_form_vectorized
        )
        from .cache import load_or_compute

        hom = homogeneity if homogeneity is not None else system.homogeneity
        if hom not in HOMOGENEITIES:
            raise ValueError(
                f"homogeneity must be one of 'translation', 'rotation', "
                f"'general'; got {shown(hom)}."
            )

        _check_c_method(c_method)

        system_identity = closed_identity = None
        if cache_path is not None:
            from ._propagator_identity import native_closed_form_identity

            system_identity = _minimal_propagator_spec(system)
            closed_identity = native_closed_form_identity(c_fn)
        native_selection = (
            c_closed_form is None
            or type(c_closed_form) is str
            and c_closed_form == "auto"
        )
        shared_cache_path = (
            cache_path
            if system_identity is not None
            and closed_identity is not None
            and native_selection
            else None
        )
        # Explicit/unversioned callbacks compute normally and never share a
        # persisted model/table payload based on repr or Python object identity.
        spec_key = {
            "system_hash": system_identity,
            "hom": hom,
            "t_max": t_max,
            "n_grid_t": n_grid_t,
            "r_max": r_max,
            "n_grid_r": n_grid_r,
            "n_grid_cos": n_grid_cos,
            "x_max": x_max,
            "n_grid_x": n_grid_x,
            "c_closed_form_repr": closed_identity,
            "c_closed_form_only": c_closed_form_only,
            "c_closed_form_vectorized": c_closed_form_vectorized,
            "interp_method": interp_method,
            "c_method": c_method,
            "c_n_gauss": int(c_n_gauss),
            "diag_C": bool(diag_C),
        }

        def _build() -> "Propagators":
            model = system.build_propagator_model(diag_C=diag_C)
            if c_fn is not None:
                cache = PropagatorCache(
                    model=model,
                    homogeneity=hom,
                    c_value_fn=c_fn,
                    interp_method=interp_method,
                    c_method=c_method,
                    n_gauss=int(c_n_gauss),
                )
            else:
                cache = PropagatorCache(
                    model=model,
                    homogeneity=hom,
                    interp_method=interp_method,
                    c_method=c_method,
                    n_gauss=int(c_n_gauss),
                )

            if c_closed_form_only:
                # Skip every spline build. C_at_batch uses
                # ``_closed_form_at_batch_diag`` from now on, which
                # routes lookups straight through the closed form.  The
                # built-in form takes arrays whatever the flag says, and the
                # per-sample loop costs it a factor of about 50.
                cache._closed_form_only = True
                cache._closed_form_vectorized = (
                    c_closed_form_vectorized or c_fn_source == "closed_form:builtin"
                )
                return cls(
                    cache=cache, homogeneity=hom, is_lazy=False, c_source=c_fn_source
                )

            if hom == "translation":
                cache.precompute_C_table_translation(
                    t_max=t_max,
                    n_grid_t=n_grid_t,
                    r_max=r_max,
                    n_grid_r=n_grid_r,
                    n_jobs=n_jobs,
                    c_method=c_method,
                    n_gauss=int(c_n_gauss),
                )
            elif hom == "rotation":
                cache.precompute_C_table_rotation(
                    t_max=t_max,
                    n_grid_t=n_grid_t,
                    n_grid_cos=n_grid_cos,
                    n_jobs=n_jobs,
                    c_method=c_method,
                    n_gauss=int(c_n_gauss),
                )
            else:  # general
                cache.precompute_C_table_general(
                    t_max=t_max,
                    n_grid_t=n_grid_t,
                    x_max=x_max,
                    n_grid_x=n_grid_x,
                    n_jobs=n_jobs,
                    c_method=c_method,
                    n_gauss=int(c_n_gauss),
                )

            # Pin lazy-cache n_jobs to 1. _LazyTimeSplineCache._build is
            # triggered from inside QMC sampling when a worker hits a new
            # parameter value; if Layer 2 (integrate_diagrams) or Layer 3
            # (Expansion.sweep) is itself parallel, an inner Parallel(...)
            # call here would spawn a nested loky pool. Lazy builds are
            # n_grid_t**2 independent _C_value_direct calls and typically
            # account for a small fraction of total wall-time, so we let
            # the outer parallelism saturate the cores instead.
            lazy_tables = [
                lazy
                for lazy in (
                    cache._lazy_translation,
                    cache._lazy_rotation,
                    cache._lazy_general,
                )
                if lazy is not None
            ]
            for lazy in lazy_tables:
                lazy.n_jobs = 1

            return cls(
                cache=cache,
                homogeneity=hom,
                is_lazy=bool(lazy_tables),
                c_source=c_fn_source or _quadrature_source(cache),
            )

        if cache_path is not None and shared_cache_path is None:
            return _build()
        return load_or_compute(
            shared_cache_path,
            spec_key,
            _build,
            operation_name="propagator table",
        )


def _resolve_closed_form_flags(
    system, c_closed_form, c_closed_form_only, c_closed_form_vectorized
):
    """:func:`_resolve_closed_form`, refusing a ``c_closed_form_only`` that
    leaves no lookup function and a vectorised contract without it."""
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
    c_fn, source = _resolve_closed_form(system, c_closed_form)
    if c_closed_form_only and c_fn is None:
        raise ValueError(
            "c_closed_form_only=True requires a closed form, but "
            "c_closed_form=None was given and this system's kernel "
            "family has no built-in one (see "
            "sft_wick.workflow.closed_forms)."
        )
    return c_fn, source


def _resolve_closed_form(system, c_closed_form):
    """``(callable | None, source label)`` for the ``c_closed_form`` setting."""
    if c_closed_form is None:
        return None, None
    if isinstance(c_closed_form, str):
        if c_closed_form != "auto":
            raise ValueError(
                f"c_closed_form must be 'auto', None or a callable; "
                f"got {shown(c_closed_form)}."
            )
        from .closed_forms import builtin_closed_form_for

        cf = builtin_closed_form_for(system)
        if cf is None:
            return None, None
        return cf, "closed_form:builtin"
    if callable(c_closed_form):
        return c_closed_form, "closed_form:user"
    raise TypeError(
        f"c_closed_form must be 'auto', None or a callable; "
        f"got {type(c_closed_form).__name__}."
    )


def _check_c_method(c_method: str) -> None:
    """Refuse a ``c_method`` that is not one of the three names."""
    if c_method not in C_METHODS:
        raise ValueError(
            f"c_method must be 'auto', 'dblquad' or 'gauss_legendre'; "
            f"got {shown(c_method)}."
        )


def _quadrature_source(cache) -> str:
    """``'quadrature:...'`` label for a cache with no closed form."""
    method = cache.c_method_resolved
    if method == "gauss_legendre":
        return f"quadrature:gauss_legendre(n={cache.n_gauss_resolved})"
    if method == "graded":
        return f"quadrature:graded(levels={cache.graded_levels_resolved})"
    if method == "auto":
        return "quadrature:per_position"
    return f"quadrature:{method}"


def propagators_from_cache(cache) -> "Propagators":
    """Wrap a hand-built :class:`~sft_wick.evaluate.PropagatorCache` as a
    :class:`Propagators`, so it can be passed to
    :meth:`~sft_wick.workflow.Expansion.evaluate` / ``sweep``.

    ``System.propagators()`` builds a cache FROM the system's own model and
    gives no supported way to use a cache that does not come from a ``System``
    at all.  The disorder-averaged
    :func:`sft_wick.spectral.spectral_cache` is exactly that: its ``R`` and
    ``C`` come from a spectrum, not from the system's ``kappa2``.

    (``Propagators`` is a plain dataclass, so a caller could always have
    constructed one by hand; this adds a supported, named entry point, not a
    new capability.)

    The component count is NOT checked here, because a cache carries no record
    of which system it was meant for.  It IS checked at use, in
    :meth:`Expansion.evaluate`, where both numbers are known.

    A cache that overrides the batch methods, or a method whose array the
    package's ``C_at_batch`` returns (``_assemble_C``, ``_lazy_lookup``,
    ``_closed_form_at_batch_diag``), may return one reused buffer, because
    the integrators copy the batches they keep.  ``C_value`` copies what an
    overridden ``_C_value_from_table`` or
    ``SpectralPropagatorCache._c_matrix`` returns, and ``C_diagonal`` what
    an overridden ``_C_diagonal_from_table`` returns.  Its batch methods
    must not write into their time arguments: every time array the package
    passes to a batch method, or on to ``_lazy_lookup``,
    ``_closed_form_at_batch_diag`` and ``_on_diagonal``, is read-only, and
    a write raises ``ValueError``.  Its ``C_at_batch``,
    ``C_value`` and ``C_diagonal`` receive copies of array positions from
    the integrators.  On a direct call as well, the package's
    ``C_at_batch`` passes copies on to an overridden ``_lazy_lookup`` or
    ``_closed_form_at_batch_diag``, and ``SpectralPropagatorCache.C_value``
    to an overridden ``_c_matrix``.  Its ``R_product`` receives a copy of
    the times dict.  ``_C_value_direct`` receives copies of its array
    arguments.

    The scalar evaluation (a zero-dimensional diagram with static couplings
    on every integrator, every diagram on ``qmc_scalar`` and ``nquad``)
    reads R through ``R_product``, which calls ``R_time``, and C through
    ``C_value`` and ``C_diagonal``.  The batched integrators evaluate every
    other diagram through the batch methods, which call ``model.R_time``
    and read the C tables, not ``R_time``, ``C_value`` or ``C_diagonal``.
    A cache that changes R or C therefore overrides the scalar accessors
    and the batch methods together.  An ``R_time`` override alone that
    doubles R gives 4 times the order-2 value of ``phi_a(x) phi_b(y)`` on
    ``qmc_scalar`` and ``nquad``, and the unchanged value on
    ``gauss_legendre``, ``qmc_vectorized`` and ``integrate_two_point_qmc``.
    """
    return Propagators(cache=cache, homogeneity="translation", is_lazy=False)


def _minimal_propagator_spec(system) -> Any:
    """Complete native content identity, or None when sharing is unsupported."""
    from ._propagator_identity import native_propagator_identity

    return native_propagator_identity(system)
